import argparse
import sys
sys.path.append('..')

import os
import json
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig

from tensorflow.keras import mixed_precision

from utils import load_data
from utils import introduce_errors 
from utils import dataset_utils

from components.losses import MaskedSparseCategoricalCrossEntropy
from components.callbacks import MyBackupAndRestore

from multiprocessing import Process, Manager

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    MAX_LENGTH = config['max_length']
    SHUFFLE_BUFFER = config['shuffle_buffer']
    BUCKET_BOUNDARIES = config['bucket_boundaries']
    BUCKET_BATCH_SIZES_PER_GPU = config['bucket_batch_sizes_per_gpu']

    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    STEPS_PER_EPOCH = config['steps_per_epoch']
    EPOCHS = config['epochs']
    USE_F16 = config['use_f16']

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']

    # loss
    LOSS = config['loss']

    # GEL config
    LANG = config['lang']
    TOKEN_FILE = config['token_file']
    TOKEN_ERR_DISTRIBUTION = config['token_err_distribution']
    CHAR_ERR_DISTRIBUTION = config['char_err_distribution']
    TOKEN_ERR_PROB = config['token_err_prob']   
    CHAR_ERR_PROB = config['char_err_prob']

    # logs
    LOG_FILE = config['log_file']
    PROFILE_BATCH = config['profile_batch']
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    BACKUP_DIR =  config['backup_dir']

    # input edits
    LABEL_PAD_VALUE = -100
    MODEL_TYPE = ""
    if MODEL in ["google/mt5-small", "google/mt5-base"]:
        MODEL_TYPE = "T5"
    else:
        MODEL_TYPE = "Bart-mine"
    print(MODEL_TYPE)


    tf.random.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)

    strategy = tf.distribute.MirroredStrategy()
    num_div = strategy.num_replicas_in_sync
    print('Number of devices: %d' % num_div)

    bucket_batch_sizes = [bucket_batch_size * num_div for bucket_batch_size in BUCKET_BATCH_SIZES_PER_GPU]

    # loading of dataset:
    manager = Manager()
    queue = manager.Queue(4 * NUM_PARALLEL)
    gel = load_data.GenereteErrorLine(
            tokens, characters, LANG, 
            TOKEN_ERR_DISTRIBUTION, CHAR_ERR_DISTRIBUTION, 
            TOKEN_ERR_PROB, CHAR_ERR_PROB)

    process = Process(
                target=load_data.data_generator, 
                args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH,))

    process.start()

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(queue.get, None),
        output_types={
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "tokenized_target_line": tf.int32,
                },
        output_shapes={
                    "input_ids": (None, ),
                    "attention_mask": (None, ),
                    "tokenized_target_line": (None, ),
                })

    def fix_format(input_batch, model_type):
        if model_type == "T5":
            dato = {
                    "input_ids": input_batch["input_ids"],
                    "attention_mask": input_batch["attention_mask"],
                    "labels": input_batch["tokenized_target_line"],
                    "decoder_input_ids": tf.concat([[0], input_batch["tokenized_target_line"][:-1]], axis=0)
                }
        elif model_type == "Bart-mine":
            dato = {
                    "input_ids": input_batch["input_ids"],
                    "attention_mask": input_batch["attention_mask"],
                    "labels": input_batch["tokenized_target_line"][1:],
                    "decoder_input_ids": input_batch["tokenized_target_line"][:-1]
                }
        elif model_type == "Bart":
            # TODO
            dato = {
                    "input_ids": input_batch["input_ids"],
                    "attention_mask": input_batch["attention_mask"],
                    "labels": input_batch["tokenized_target_line"],
                    "decoder_input_ids": tf.concat([[0], input_batch["tokenized_target_line"][:-1]], axis=0)
                }

        return dato

    dataset = dataset.map(lambda input_batch: fix_format(input_batch, MODEL_TYPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x['input_ids'])[0],
            bucket_boundaries=BUCKET_BOUNDARIES,
            bucket_batch_sizes=bucket_batch_sizes
    )
    if LABEL_PAD_VALUE:
        dataset = dataset.map(lambda x, y: (x, dataset_utils.change_value(y, 0, LABEL_PAD_VALUE)))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if USE_F16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    with strategy.scope():
        if OPTIMIZER_NAME == 'Adam':
            optimizer = tf.keras.optimizers.Adam(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'AdamW':
            optimizer = tf.keras.optimizers.experimental.AdamW(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'Adafactor':
            optimizer = tf.keras.optimizers.experimental.Adafactor(**OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'AdaptiveAdam':
            class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, d_model):
                    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                    self.d_model = tf.cast(d_model, tf.float32)

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    lr = (1.0/tf.math.sqrt(self.d_model)) * tf.math.minimum(1.0 / tf.math.sqrt(step), (1.0 / tf.math.sqrt(self.warmup_steps)) * ((1.0 * step) / self.warmup_steps))
                    return lr
            learning_rate = LRSchedule(OPTIMIZER_PARAMS['warmup_steps'], MAX_LENGTH)
            del OPTIMIZER_PARAMS['warmup_steps']
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, **OPTIMIZER_PARAMS)
        elif OPTIMIZER_NAME == 'CosineDecay':
            # OPTIMIZER_PARAMS['initial_learning_rate'], OPTIMIZER_PARAMS['decay_steps']
            cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(**OPTIMIZER_PARAMS)
            optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate=cosine_decay_scheduler)


    with strategy.scope(): 
        loss = None   
        if LOSS == "SCC":
            loss = MaskedSparseCategoricalCrossEntropy()


    with strategy.scope():
        if FROM_CONFIG:
            config = AutoConfig.from_pretrained(MODEL)
            model = TFAutoModelForSeq2SeqLM.from_config(config)
        else:
            print("Use pretrained model...")
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    ### Callbacks

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINT_PATH, 'ckpt-{epoch}/'),
        save_weights_only=True,
        save_freq="epoch")

    mybackup = MyBackupAndRestore(BACKUP_DIR, optimizer, model)
    status = mybackup.checkpoint.restore(mybackup.manager.latest_checkpoint)
    print("STATUS:", status)
    initial_epoch = mybackup._ckpt_saved_epoch
    print("INITIAL EPOCH:", int(initial_epoch))

    profiler = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_FILE, 
        profile_batch=PROFILE_BATCH)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_FILE, 
        histogram_freq=1)

    callbacks = [
        model_checkpoint,
        mybackup,
        profiler,
        tensorboard_callback
    ]

    ### Train

    if USE_F16:
        model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)

    if STEPS_PER_EPOCH:
        model.fit(
            dataset, 
            initial_epoch=int(initial_epoch),
            callbacks=callbacks, 
            epochs=EPOCHS, 
            steps_per_epoch=STEPS_PER_EPOCH)
    else:
        model.fit(
            dataset,
            initial_epoch=int(initial_epoch),
            callbacks=callbacks, 
            epochs=EPOCHS)
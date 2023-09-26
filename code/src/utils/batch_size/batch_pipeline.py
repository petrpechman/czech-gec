import sys
sys.path.append('..')

import os
import json
import dataset_utils 
import tensorflow as tf

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM
from tensorflow.keras import mixed_precision
from components.losses import MaskedSparseCategoricalCrossEntropy


def main(batch_size: int, max_length: int, epochs:int, steps_per_epoch:int, config: str, filename: str):
    BATCH_SIZE = batch_size
    MAX_LENGTH = max_length
    EPOCHS=epochs
    STEPS_PER_EPOCH=steps_per_epoch
    CONFIG = config
    FILENAME = filename

    with open(CONFIG) as json_file:
        config = json.load(json_file)

    SEED = config['seed']

    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    USE_F16 = config['use_f16']

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']

    # loss
    LOSS = config['loss']

    # input edits
    LABEL_PAD_VALUE = -100
    MODEL_TYPE = ""
    if MODEL in ["google/mt5-small", "google/mt5-base"]:
        MODEL_TYPE = "T5"
    else:
        MODEL_TYPE = "Bart-mine"
    # print(MODEL_TYPE)

    tf.random.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    strategy = tf.distribute.MirroredStrategy()
    num_div = strategy.num_replicas_in_sync
    # print('Number of devices: %d' % num_div)

    def get_tokenized_sentences(line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, text_target=line, max_length=max_length, padding='max_length', truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask'], tokenized['labels']

    def tokenize_line(line, max_length):
        input_ids, attention_mask, labels = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32, tf.int32])
        dato = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'tokenized_target_line': labels[0]
        }
        return dato

    def ensure_shapes(input_dict, max_length):
        return {key: tf.ensure_shape(val, (max_length)) for key, val in input_dict.items()}

    dataset = tf.data.TextLineDataset([FILENAME])
    dataset = dataset.map(lambda line: tokenize_line(line, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_batch: dataset_utils.fix_format(input_batch, MODEL_TYPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_dict: ensure_shapes(input_dict, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: dataset_utils.change_value(x, y, 0, LABEL_PAD_VALUE, MODEL_TYPE))
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
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    if USE_F16:
        ... # already fixed
        # model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        # model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)

    model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0)

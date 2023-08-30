import sys
sys.path.append('..')

import os
import time
import shutil
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from m2scorer.levenshtein import batch_multi_pre_rec_f1_part, batch_multi_pre_rec_f1
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer

from utils.time_check import timeout

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)
    
    SEED = config['seed']

    # data loading
    M2_DATA = config['m2_data']
    MAX_LENGTH = config['max_length']
    BATCH_SIZE = config['batch_size']
    
    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    USE_F16 = config['use_f16']
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']

    # evaluation
    MAX_UNCHANGED_WORDS = config['max_unchanged_words']
    BETA = config['beta']
    IGNORE_WHITESPACE_CASING = config['ignore_whitespace_casing']
    VERBOSE = config['verbose']
    VERY_VERBOSE = config['very_verbose']

    TIMEOUT = config['timeout']
    
    tf.random.set_seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']
    
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        if OPTIMIZER_NAME == 'AdaptiveAdam':
            class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, d_model):
                    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                    self.d_model = tf.cast(d_model, tf.float32)

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    lr = (1.0/tf.math.sqrt(self.d_model)) * tf.math.minimum(1.0 / tf.math.sqrt(step), (1.0 / tf.math.sqrt(self.warmup_steps)) * ((1.0 * step) / self.warmup_steps))
                    return lr

            lr = LRSchedule(OPTIMIZER_PARAMS['warmup_steps'], MAX_LENGTH)
            beta1 = OPTIMIZER_PARAMS['beta1']
            beta2 = OPTIMIZER_PARAMS['beta2']
            epsilon = OPTIMIZER_PARAMS['epsilon']
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                clipvalue=OPTIMIZER_PARAMS['clipvalue'], 
                global_clipnorm=OPTIMIZER_PARAMS['global_clipnorm'],
                beta_1=beta1,
                beta_2=beta2,
                epsilon=epsilon)
    
    # loading of dataset:
    def get_tokenized_sentences(line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, max_length=MAX_LENGTH, truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask']

    def tokenize_line(line):
        input_ids, attention_mask = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32])
        dato = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
        }
        return dato

    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA)
    #OK
    dataset = tf.data.Dataset.from_tensor_slices((dev_source_sentences))
    dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #OK
    dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #OK

    # MYBACKUP:
    # config = AutoConfig.from_pretrained(MODEL)
    # model = TFAutoModelForSeq2SeqLM.from_config(config)

    # save_path = "../bart_find/tmp/mybackup/ckpt-1/"
    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # status = checkpoint.restore(save_path).expect_partial()



    # INFERENCE:
    BACKUP_DIR = "../bart_find/tmp/backup/"
    MODEL = "../../models/transformer/"
    MODEL_CHECKPOINT_PATH = "../bart_find/tmp/checkpoint"
    unevaluated_checkpoint = 'ckpt-15'

    config = AutoConfig.from_pretrained(MODEL)
    model = TFAutoModelForSeq2SeqLM.from_config(config)

    _ckpt_saved_epoch = tf.Variable(
        initial_value=tf.constant(0, dtype=tf.int64), 
        name="ckpt_saved_epoch"
    )

    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer, 
        model=model,
        ckpt_saved_epoch=_ckpt_saved_epoch,
    )

    manager_ch = tf.train.CheckpointManager(
                checkpoint, 
                BACKUP_DIR, 
                max_to_keep=2)

    status = checkpoint.restore(manager_ch.latest_checkpoint).expect_partial()
    print("STATUS:", status)

    # model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()

    # model.load_weights(BACKUP_DIR).expect_partial()

    # BACKUP:
    # callback = tf.keras.callbacks.BackupAndRestore(backup_dir=BACKUP_DIR)

    # config = AutoConfig.from_pretrained(MODEL)
    # model = TFAutoModelForSeq2SeqLM.from_config(config)
    # model.predict(x=None,callbacks=[callback])

    # model = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

    # config = AutoConfig.from_pretrained(MODEL)
    # model = TFAutoModelForSeq2SeqLM.from_config(config)
    # checkpoint = tf.train.Checkpoint(model=model)
    # checkpoint.restore(tf.train.latest_checkpoint(BACKUP_DIR))

    for i, batch in enumerate(dataset):
        print(f"Generate {i+1}. batch.") 
        preds = model.generate(batch['input_ids'], max_length=150)
        batch_sentences = tokenizer.batch_decode(preds)
        break

    print(batch['input_ids'])
    print(preds)
    print(batch_sentences)
    # print(model.generate([batch['input_ids'][0]]))
    


if __name__ == '__main__':
    main("../bart_find/config.json")
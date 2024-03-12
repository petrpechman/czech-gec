import sys
sys.path.append('..')

import os
import json
import tensorflow as tf

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM

from tensorflow.keras import mixed_precision

from utils import load_data
from utils import dataset_utils
from utils import introduce_errors
from utils import create_errors

from utils.components.callbacks import MyBackupAndRestore
from utils.components.losses import MaskedSparseCategoricalCrossEntropy

from multiprocessing import Process, Manager


def main(config_filename: str):

    with open(config_filename) as json_file:
        config = json.load(json_file)

    with open(config['errors_config']) as json_file:
        errors_config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    MAX_LENGTH = config['max_length']
    SHUFFLE_BUFFER = config['shuffle_buffer']
    BUCKET_BOUNDARIES = config['bucket_boundaries']
    BUCKET_BATCH_SIZES_PER_GPU = config['bucket_batch_sizes_per_gpu']
    # data from file
    ERRORS_FROM_FILE = config.get('errors_from_file', False)
    REVERTED_PIPELINE = config.get('reverted_pipeline', False)

    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config'] # means from scratch
    STEPS_PER_EPOCH = config['steps_per_epoch']
    EPOCHS = config['epochs']
    USE_F16 = config['use_f16']

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']
    LR = OPTIMIZER_PARAMS.get('learning_rate', None)

    # loss
    LOSS = config['loss']

    # GEL config
    LANG = config['lang']
    TOKEN_FILE = config['token_file']
    TOKEN_ERR_DISTRIBUTION = config['token_err_distribution']
    DERINET_DIST = config['derinet_dist']
    CHAR_ERR_DISTRIBUTION = config['char_err_distribution']
    TOKEN_ERR_PROB = config['token_err_prob']   
    CHAR_ERR_PROB = config['char_err_prob']

    # logs
    LOG_FILE = config['log_file']
    PROFILE_BATCH = config['profile_batch']
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    BACKUP_DIR =  config['backup_dir']
    COUNT_OUTPUT = config.get('count_output', None)

    # mixture of datasets:
    MIXTURE_DATASET_PATHS = config.get('mixture_dataset_paths', None)
    RATIO_MIX = config.get('ratio_mix', [2, 1]) # first is main pipeline

    # input edits
    LABEL_PAD_VALUE = -100
    MODEL_TYPE = ""
    if MODEL in ["google/mt5-small", "google/mt5-base", "google/mt5-large"]:
        MODEL_TYPE = "T5"
    else:
        MODEL_TYPE = "Bart-mine"
    print(MODEL_TYPE)

    ### Init 
    tf.random.set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)
    strategy = tf.distribute.MirroredStrategy()
    num_div = strategy.num_replicas_in_sync
    print('Number of devices: %d' % num_div)
    bucket_batch_sizes = [bucket_batch_size * num_div for bucket_batch_size in BUCKET_BATCH_SIZES_PER_GPU]
    print("Bucket batch size: ", bucket_batch_sizes)
    if REVERTED_PIPELINE:
        print('It is used REVERTED_PIPELINE.')
    ###

    ### Dataset loading:
    manager = Manager()
    queue = manager.Queue(4 * NUM_PARALLEL)
    if not ERRORS_FROM_FILE:
        error_generator = create_errors.ErrorGenerator(
            errors_config, tokens, characters, 
            CHAR_ERR_DISTRIBUTION, CHAR_ERR_PROB, 0.01,
            TOKEN_ERR_DISTRIBUTION, TOKEN_ERR_PROB, 0.2,
            DERINET_DIST)
        gel = None
        # gel = load_data.GenereteErrorLine(
        #     tokens, characters, LANG, 
        #     TOKEN_ERR_DISTRIBUTION, CHAR_ERR_DISTRIBUTION, 
        #     TOKEN_ERR_PROB, CHAR_ERR_PROB)
    else:
        gel = None
        error_generator = None

    # main process that creates pool, goes over possible files and manage other read processes
    process = Process(
                target=load_data.data_generator, 
                args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH, 
                      ERRORS_FROM_FILE, REVERTED_PIPELINE, error_generator, LANG, 
                      COUNT_OUTPUT, ))

    process.start()

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(queue.get, None),
        output_types={
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "tokenized_target_line": tf.int32,
                    "original_sentence": tf.string,
                    "correct_sentence": tf.string,
                },
        output_shapes={
                    "input_ids": (None, ),
                    "attention_mask": (None, ),
                    "tokenized_target_line": (None, ),
                    "original_sentence": (),
                    "correct_sentence": (),
                })

    dataset = dataset.map(lambda input_batch: dataset_utils.fix_format(input_batch, MODEL_TYPE), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if MIXTURE_DATASET_PATHS:
        ### Dataset akces:
        num_parallel_akces = 2
        manager_akces = Manager()
        queue_akces = manager_akces.Queue(4 * num_parallel_akces)
        gel_akces = None
        error_generator_akces = None

        process_akces = Process(
                    target=load_data.data_generator, 
                    args=(queue_akces, MIXTURE_DATASET_PATHS, num_parallel_akces, 
                          gel_akces, tokenizer, MAX_LENGTH, 
                          True, REVERTED_PIPELINE, error_generator_akces, LANG, 
                          COUNT_OUTPUT, ))
        process_akces.start()
        dataset_akces = tf.data.Dataset.from_generator(
            lambda: iter(queue_akces.get, None),
            output_types={
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "tokenized_target_line": tf.int32,
                        "original_sentence": tf.string,
                        "correct_sentence": tf.string,
                    },
            output_shapes={
                        "input_ids": (None, ),
                        "attention_mask": (None, ),
                        "tokenized_target_line": (None, ),
                        "original_sentence": (),
                        "correct_sentence": (),
                    })
        dataset_akces = dataset_akces.map(lambda input_batch: dataset_utils.fix_format(input_batch, MODEL_TYPE), 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset_akces = dataset_akces.map(dataset_utils.split_features_and_labels, 
                                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ### Mixture:
        r1 = RATIO_MIX[0] # 2
        r2 = RATIO_MIX[1] # 1
        b1 = dataset.ragged_batch(r1)
        b2 = dataset_akces.ragged_batch(r2)
        zipped = tf.data.Dataset.zip((b1, b2)).map(dataset_utils.merge_ragged_batches, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        zipped = zipped.unbatch() # lze mozna nahradit s .rebatch(1)
        zipped = zipped.batch(1)
        zipped = zipped.map(dataset_utils.retype, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = zipped.unbatch()
        ###

    # dataset = dataset.shuffle(SHUFFLE_BUFFER)
    # dataset = dataset.bucket_by_sequence_length(
    #         element_length_func=lambda x, y: tf.shape(x['input_ids'])[0], # zde asi chyba
    #         bucket_boundaries=BUCKET_BOUNDARIES,
    #         bucket_batch_sizes=bucket_batch_sizes
    # )
    # dataset = dataset.map(lambda x, y: dataset_utils.change_value(x, y, 0, LABEL_PAD_VALUE, MODEL_TYPE))
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    for d in dataset:
        break
    print(d)
    
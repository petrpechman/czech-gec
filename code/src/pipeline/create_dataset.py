
import sys
sys.path.append('..')

import json
import tensorflow as tf

from transformers import AutoTokenizer

from utils import load_data
from utils import introduce_errors

from multiprocessing import Process, Manager


def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    MAX_LENGTH = config['max_length']
    # data from file
    ERRORS_FROM_FILE = False
    REVERTED_PIPELINE = False

    # model
    TOKENIZER = config['tokenizer']

    # GEL config
    LANG = config['lang']
    TOKEN_FILE = config['token_file']
    TOKEN_ERR_DISTRIBUTION = config['token_err_distribution']
    CHAR_ERR_DISTRIBUTION = config['char_err_distribution']
    TOKEN_ERR_PROB = config['token_err_prob']   
    CHAR_ERR_PROB = config['char_err_prob']

    DATASET_FILEPATH = config['dataset_filepath']

    ### Init 
    tf.random.set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)
    # bucket_batch_sizes = [bucket_batch_size * num_div for bucket_batch_size in BUCKET_BATCH_SIZES_PER_GPU]
    ###

    ### Dataset loading:
    manager = Manager()
    queue = manager.Queue(4 * NUM_PARALLEL)
    if not ERRORS_FROM_FILE:
        gel = load_data.GenereteErrorLine(
            tokens, characters, LANG, 
            TOKEN_ERR_DISTRIBUTION, CHAR_ERR_DISTRIBUTION, 
            TOKEN_ERR_PROB, CHAR_ERR_PROB)
    else:
        gel = None

    # main process that creates pool, goes over possible files and manage other read processes
    process = Process(
                target=load_data.data_generator, 
                args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH, ERRORS_FROM_FILE, REVERTED_PIPELINE, ))

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
    
    print("Predicting...")
    for i, batch in enumerate(dataset):
        with open(DATASET_FILEPATH, "a+") as file:
            line = batch['correct_sentence'].decode("utf-8") + "\t" + batch['original_sentence'].decode("utf-8") + "\n"
            file.write(line)

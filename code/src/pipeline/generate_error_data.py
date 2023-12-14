import sys
sys.path.append('..')

import os
import time
import json
import shutil
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)
    ### Params:
    num_beams = 5
    min_length = 0
    length_penalty = 1.0
    diversity_penalty = 5.0
    ###
    
    SEED = config['seed']

    # data loading
    DATA_FILEPATH = config.get('data_filepath', None) # tokenized data by udpipe tokenizer
    if not DATA_FILEPATH:
        print("Need to specify filepath...")
        return
    
    BATCH_SIZE = config['batch_size']
    
    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    USE_F16 = False
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    MAX_EVAL_LENGTH = config['max_eval_length']
    FILE_PREDICTIONS = 'predictions.txt'

    MODEL_TYPE = ""
    if MODEL in ["google/mt5-small", "google/mt5-base"]:
        MODEL_TYPE = "T5"
    else:
        MODEL_TYPE = "Bart-mine"
    print(MODEL_TYPE)
    
    tf.random.set_seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    
    ### Dataset loadings:
    def get_tokenized_sentences(line):
        # only tokenize line
        line = line.decode('utf-8')
        tokenized = tokenizer(line, max_length=MAX_EVAL_LENGTH, truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask']

    def tokenize_line(line):
        # wrapper for tokenize_line
        input_ids, attention_mask = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32])
        dato = {'input_ids': input_ids[0],
                'attention_mask': attention_mask[0]}
        return dato
    
    def get_dataset_pipeline(data_filepath) -> tf.data.Dataset:
        dataset = tf.data.TextLineDataset([data_filepath], num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    dataset = get_dataset_pipeline(DATA_FILEPATH)
    ###
    
    ### Prepare right model:
    if USE_F16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

    with strategy.scope():
        if FROM_CONFIG:
            config = AutoConfig.from_pretrained(MODEL)
            model = TFAutoModelForSeq2SeqLM.from_config(config)
        else:
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

    if USE_F16 and MODEL_TYPE == "Bart-mine":
        model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)
    ###

    # prepare udpipe tokenizer
    udpipe_tokenizer = UDPipeTokenizer("cs")
    
    def generate_and_score(unevaluated_checkpoint, dataset, predictions_file):
        step = int(unevaluated_checkpoint[5:])
        predictions_filepath = os.path.join(MODEL_CHECKPOINT_PATH, str(step) + "-" + predictions_file)

        ### Load model weights for evaluation
        model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()
        ###

        print("Predicting...")
        for i, batch in enumerate(dataset):
            print(f"Generate {i+1}. batch.") 
            with open(predictions_filepath, "a+") as file:
                preds = model.generate(
                    batch['input_ids'], 
                    max_length=MAX_EVAL_LENGTH,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    diversity_penalty=diversity_penalty,
                    )
                batch_sentences = tokenizer.batch_decode(preds, skip_special_tokens=True)
                print("Write into file...")
                for i, line in enumerate(batch_sentences):
                    tokenization = udpipe_tokenizer.tokenize(line)
                    sentence = " ".join([token.string for tokens_of_part in tokenization for token in tokens_of_part]) if len(tokenization) > 0 else ""
                    file.write(sentence + '\n')
        print("End of predicting...")

    while True:
        if os.path.isdir(MODEL_CHECKPOINT_PATH):
            unevaluated = [f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.startswith('ckpt')]
            unevaluated = sorted(unevaluated)
            
            for unevaluated_checkpoint in unevaluated:
                try:
                    generate_and_score(unevaluated_checkpoint, dataset, FILE_PREDICTIONS)
                    
                    # print(f"Delete: {os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint)}")
                    # shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint))
                except:
                    print("Something went wrong... Try again...")
            
            break        
        time.sleep(10)

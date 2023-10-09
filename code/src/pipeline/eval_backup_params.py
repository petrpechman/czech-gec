import sys
sys.path.append('..')

import os
import tensorflow as tf

import argparse

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from m2scorer.levenshtein import batch_multi_pre_rec_f1_part
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer

from utils.components.callbacks import MyBackupAndRestore
from utils.components.losses import MaskedSparseCategoricalCrossEntropy

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)

    ###
    num_beams = 4
    min_length = 0
    length_penalty = 1.0
    specific_folder = f"nb-{num_beams}-minl-{min_length}-lp-{length_penalty}"
    dump_folder = os.path.join('dumps', specific_folder)
    os.makedirs(dump_folder, exist_ok=True)
    ###
    
    SEED = config['seed']

    # data loading
    M2_DATA_DEV = config['m2_data_dev']
    M2_DATA_TEST = config['m2_data_test']
    MAX_LENGTH = config['max_length']
    BATCH_SIZE = config['batch_size']
    
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
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    BACKUP_DIR = config['backup_dir']

    # evaluation
    MAX_UNCHANGED_WORDS = config['max_unchanged_words']
    BETA = config['beta']
    IGNORE_WHITESPACE_CASING = config['ignore_whitespace_casing']
    VERBOSE = config['verbose']
    VERY_VERBOSE = config['very_verbose']
    
    MAX_EVAL_LENGTH = config['max_eval_length']

    # OUTPUT_DIR = 'results' # "m2_data": "../../data/geccc/dev/sorted_sentence.m2",
    OUTPUT_DIR_DEV = f"results-geccc-dev-{specific_folder}" # "m2_data": "../../data/akces-gec/dev/dev.all.m2",
    OUTPUT_DIR_TEST = f"results-geccc-test-{specific_folder}" # "m2_data": "../../data/akces-gec/test/test.all.m2",
    
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
    
    def get_dataset_pipeline(dev_source_sentences) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((dev_source_sentences))
        dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA_DEV)
    test_source_sentences, test_gold_edits = load_annotation(M2_DATA_TEST)

    dev_dataset = get_dataset_pipeline(dev_source_sentences)
    test_dataset = get_dataset_pipeline(test_source_sentences)
    ###

    udpipe_tokenizer = UDPipeTokenizer("cs")

    def compute_metrics(tokenized_predicted_sentences, dev_source_sentences, dev_gold_edits):
        total_stat_correct, total_stat_proposed, total_stat_gold = 0, 0, 0 
        size = BATCH_SIZE
        for i in range(0, len(tokenized_predicted_sentences), size):
            ###
            print("Batch of sentences:")
            for s in tokenized_predicted_sentences[i:i+size]:
                print(s)
            print("End of batch")
            ###
            stat_correct, stat_proposed, stat_gold = batch_multi_pre_rec_f1_part(
                tokenized_predicted_sentences[i:i+size], 
                dev_source_sentences[i:i+size], 
                dev_gold_edits[i:i+size],
                MAX_UNCHANGED_WORDS, BETA, IGNORE_WHITESPACE_CASING, VERBOSE, VERY_VERBOSE)
            total_stat_correct += stat_correct
            total_stat_proposed += stat_proposed
            total_stat_gold += stat_gold
            p  = total_stat_correct / total_stat_proposed if total_stat_proposed > 0 else 0
            r  = total_stat_correct / total_stat_gold if total_stat_gold > 0 else 0
            f1 = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0
            print(f"Step {i+1}")
            print("Precision:\t", p)
            print("Recall:\t", r)
            print("F1:\t", f1)
        return total_stat_correct, total_stat_proposed, total_stat_gold

    ###
    if USE_F16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

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
            print("Use pretrained model...")
            model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    def generate_and_score(mybackup, dataset, source_sentences, gold_edits, output_dir, tag):
        status = mybackup.checkpoint.restore(mybackup.manager.latest_checkpoint).expect_partial()
        print("STATUS:", status)
        step = mybackup._ckpt_saved_epoch
        print("INITIAL EPOCH:", int(step))

        result_dir = os.path.join(MODEL_CHECKPOINT_PATH, output_dir)

        print("Generating...")
        predicted_sentences = []
        for i, batch in enumerate(dataset):
            print(f"Generate {i+1}. batch.") 
            preds = model.generate(
                batch['input_ids'], 
                max_length=MAX_EVAL_LENGTH,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                )
            batch_sentences = tokenizer.batch_decode(preds, skip_special_tokens=True)
            predicted_sentences = predicted_sentences + batch_sentences
        print("End of generating...")

        print("Udpipe tokenization...")
        tokenized_predicted_sentences = []
        for i, line in enumerate(predicted_sentences):
            if i % BATCH_SIZE == 0:
                print(f"Tokenize {i+BATCH_SIZE} sentences.")
            tokenization = udpipe_tokenizer.tokenize(line)
            sentence = " ".join([token.string for token in tokenization[0]]) if len(tokenization) > 0 else ""
            tokenized_predicted_sentences.append(sentence)
        print("End of tokenization...")

        print("Compute metrics...")
        total_stat_correct, total_stat_proposed, total_stat_gold = compute_metrics(tokenized_predicted_sentences, source_sentences, gold_edits)
        print("End of computing...")

        print("Create dump...")
        with open(os.path.join(dump_folder, f"errors-{tag}.txt"), 'w') as file:
            source_sentences_w = [sentence + "\n" for sentence in source_sentences]
            file.writelines(source_sentences_w)
        with open(os.path.join(dump_folder, f"corrected-{tag}.txt"), 'w') as file:
            tokenized_predicted_sentences_w = [sentence + "\n" for sentence in tokenized_predicted_sentences]
            file.writelines(tokenized_predicted_sentences_w)
        print("End of creating dump...")

        print("Write metrics into files...")
        p  = total_stat_correct / total_stat_proposed if total_stat_proposed > 0 else 0
        r  = total_stat_correct / total_stat_gold if total_stat_gold > 0 else 0
        f1 = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0
        file_writer = tf.summary.create_file_writer(result_dir)
        with file_writer.as_default():
            tf.summary.scalar('epoch_precision', p, step)
            tf.summary.scalar('epoch_recall', r, step)
            tf.summary.scalar('epoch_f1', f1, step)
            text = "  \n".join(tokenized_predicted_sentences[0:40])
            print(text)
            tf.summary.text("predictions", text, step)

    try:
        mybackup = MyBackupAndRestore(BACKUP_DIR, optimizer, model)
        generate_and_score(mybackup, dev_dataset, dev_source_sentences, dev_gold_edits, OUTPUT_DIR_DEV, 'dev-geccc')
        generate_and_score(mybackup, test_dataset, test_source_sentences, test_gold_edits, OUTPUT_DIR_TEST, 'test-geccc')
    except Exception as e:
        print(e)
        print("Something went wrong... Try again...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    args = parser.parse_args()

    main(args.config)
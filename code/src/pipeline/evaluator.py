import sys
sys.path.append('..')

import os
import time
import shutil
import errant
import numpy as np
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from m2scorer.levenshtein import batch_multi_pre_rec_f1_part
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import dataset_utils
from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer

from collections import Counter
from errant.commands.compare_m2 import simplify_edits, process_edits, evaluate_edits, merge_dict

def noop_edit(id: int = 0):
    result = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||" + str(id)
    return result

def create_m2(annotator, source_sentence, predicted_sentence):
    orig = source_sentence
    cor = predicted_sentence
    cor_id = 0

    lev = False
    merge = "all-split"

    orig = annotator.parse(orig)
    output = " ".join(["S"] + [token.text for token in orig]) + "\n"

    cor = cor.strip()
    if orig.text.strip() == cor:
        output = output + noop_edit(cor_id) + "\n"
    else:
        cor = annotator.parse(cor)
        edits = annotator.annotate(orig, cor, lev, merge)
        for edit in edits:
            output = output + edit.to_m2(cor_id) + "\n"
    
    return output.strip()

def main(config_filename: str):
    with open(config_filename) as json_file:
        config = json.load(json_file)
    ### Params:
    num_beams = 4
    min_length = 0
    length_penalty = 1.0
    ###
    
    SEED = config['seed']

    # data loading
    EVAL_TYPE_DEV, EVAL_TYPE_TEST = ['m2_scorer'], ['m2_scorer']
    M2_DATA_DEV = config['m2_data_dev']
    if not isinstance(M2_DATA_DEV, str):
        M2_DATA_DEV, EVAL_TYPE_DEV = M2_DATA_DEV

    M2_DATA_TEST = config['m2_data_test']
    if not isinstance(M2_DATA_TEST, str):
        M2_DATA_TEST, EVAL_TYPE_TEST = M2_DATA_TEST  
    OTHER_DATASETS = config.get('other_datasets', [])
    BATCH_SIZE = config['batch_size']
    
    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
    FROM_CONFIG = config['from_config']
    # USE_F16 = config['use_f16']
    USE_F16 = False
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']

    # evaluation
    MAX_UNCHANGED_WORDS = config['max_unchanged_words']
    BETA = config['beta']
    IGNORE_WHITESPACE_CASING = config['ignore_whitespace_casing']
    VERBOSE = config['verbose']
    VERY_VERBOSE = config['very_verbose']
    
    MAX_EVAL_LENGTH = config['max_eval_length']

    # TIMEOUT = config['timeout'] # it cat be useful for geccc

    # OUTPUT_DIR = 'results' # "m2_data": "../../data/geccc/dev/sorted_sentence.m2",
    OUTPUT_DIR_DEV = 'results-dev' # "m2_data": "../../data/akces-gec/dev/dev.all.m2",
    OUTPUT_DIR_TEST = 'results-test' # "m2_data": "../../data/akces-gec/test/test.all.m2",
    FILE_DEV_PREDICTIONS = 'predictions_dev.txt'
    FILE_TEST_PREDICTIONS = 'predictions_test.txt'

    BEST_CKPT_FILENAME = config.get("best_ckpt_filename", None)
    if BEST_CKPT_FILENAME:
        with open(BEST_CKPT_FILENAME) as json_file:
            best_ckpt = json.load(json_file)
        BEST_CKPT_NAME = best_ckpt['name']
        BEST_CKPT_FSCORE = best_ckpt['fscore']

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
    
    def get_dataset_pipeline(source_sentences) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices((source_sentences))
        dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(dataset_utils.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    
    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA_DEV)
    test_source_sentences, test_gold_edits = load_annotation(M2_DATA_TEST)

    dev_ref_m2, test_ref_m2 = None, None
    if 'errant' in EVAL_TYPE_DEV:
        dev_ref_m2 = open(M2_DATA_DEV).read().strip().split("\n\n")
    if 'errant' in EVAL_TYPE_TEST:
        test_ref_m2 = open(M2_DATA_TEST).read().strip().split("\n\n")
    
    datasets = []
    refs = []
    eval_types = []
    for dataset in OTHER_DATASETS:
        if not isinstance(dataset, str):
            dataset, eval_type = dataset
        else:
            eval_type = ['m2_scorer'] 
        source_sentences, gold_edits = load_annotation(dataset)
        if 'errant' in eval_type:
            ref_m2 = open(dataset).read().strip().split("\n\n")
            refs.append(ref_m2)
        else:
            refs.append(None)
        eval_types.append(eval_type)
        datasets.append((source_sentences, gold_edits, dataset))

    dev_dataset = get_dataset_pipeline(dev_source_sentences)
    test_dataset = get_dataset_pipeline(test_source_sentences)
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

    if USE_F16:
        model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)
    ###

    # prepare udpipe tokenizer
    udpipe_tokenizer = UDPipeTokenizer("cs")

    def compute_metrics_m2scorer(tokenized_predicted_sentences, source_sentences, gold_edits):
        '''
        Goes through predicted sentences and computes true positives (stat_correct), 
        TP+FN (stat_gold), TP+FP (stat_proposed) for every batch.
        Finally it computes precision, recall and f score. 
        '''
        total_stat_correct, total_stat_proposed, total_stat_gold = 0, 0, 0 
        size = BATCH_SIZE
        for i in range(0, len(tokenized_predicted_sentences), size):
            # batch_multi_pre_rec_f1_part is created by petr pechman in fork from M2scorer
            # (https://github.com/petrpechman/m2scorer/blob/cbf794b370be2fc77f98ee9531cf33001572b7ce/m2scorer/levenshtein.py#L866),
            # it is almost same as batch_multi_pre_rec_f1
            stat_correct, stat_proposed, stat_gold = batch_multi_pre_rec_f1_part(
                tokenized_predicted_sentences[i:i+size], 
                source_sentences[i:i+size], 
                gold_edits[i:i+size],
                MAX_UNCHANGED_WORDS, BETA, IGNORE_WHITESPACE_CASING, VERBOSE, VERY_VERBOSE)
            total_stat_correct += stat_correct
            total_stat_proposed += stat_proposed
            total_stat_gold += stat_gold
            p  = total_stat_correct / total_stat_proposed if total_stat_proposed > 0 else 0
            r  = total_stat_correct / total_stat_gold if total_stat_gold > 0 else 0
            f_score = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0
            print(f"Step {i+1}")
            print("Precision:\t", p)
            print("Recall:\t", r)
            print("F-Score:\t", f_score)
        return total_stat_correct, total_stat_proposed, total_stat_gold
    
    def generate_and_score(unevaluated_checkpoint, dataset, source_sentences, gold_edits, output_dir, predictions_file,
                           ref_m2, eval_type) -> float:
        step = int(unevaluated_checkpoint[5:])
        result_dir = os.path.join(MODEL_CHECKPOINT_PATH, output_dir)
        predictions_filepath = os.path.join(MODEL_CHECKPOINT_PATH, str(step) + "-" + predictions_file)

        ### Load model weights for evaluation
        model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()
        ###

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
            sentence = " ".join([token.string for tokens_of_part in tokenization for token in tokens_of_part]) if len(tokenization) > 0 else ""
            tokenized_predicted_sentences.append(sentence)
        print("End of tokenization...")

        file_writer = tf.summary.create_file_writer(result_dir)
        if 'm2_scorer' in eval_type:
            print("Compute metrics m2 scorer...")
            total_stat_correct, total_stat_proposed, total_stat_gold = compute_metrics_m2scorer(tokenized_predicted_sentences, source_sentences, gold_edits)
            p  = total_stat_correct / total_stat_proposed if total_stat_proposed > 0 else 0
            r  = total_stat_correct / total_stat_gold if total_stat_gold > 0 else 0
            f_score = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0
            print("End of computing m2 scorer...")

            print("Write into files...")
            with file_writer.as_default():
                tf.summary.scalar('epoch_m2scorer_precision', p, step)
                tf.summary.scalar('epoch_m2scorer_recall', r, step)
                tf.summary.scalar('epoch_m2scorer_f_score', f_score, step)
            print("End of writing into files...")
        if 'errant' in eval_type:
            hyp_m2 = []
            annotator = errant.load('cs')
            for source_sentence, tokenized_predicted_sentence in zip(source_sentences, tokenized_predicted_sentences):
                m2_sentence = create_m2(annotator, source_sentence, tokenized_predicted_sentence)
                hyp_m2.append(m2_sentence)
            
            best_dict = Counter({"tp":0, "fp":0, "fn":0})
            best_cats = {}
            # Process each sentence
            sents = zip(hyp_m2, ref_m2)
            for sent_id, sent in enumerate(sents):
                # Simplify the edits into lists of lists
                hyp_edits = simplify_edits(sent[0])
                ref_edits = simplify_edits(sent[1])
                # Process the edits for detection/correction based on args
                class Args:
                    def __init__(self) -> None:
                        self.beta = BETA
                        self.dt = None
                        self.ds = None
                        self.single = None
                        self.multi = None
                        self.filt = None
                        self.cse = None
                        self.verbose = None
                args = Args()
                hyp_dict = process_edits(hyp_edits, args)
                ref_dict = process_edits(ref_edits, args)
                # Evaluate edits and get best TP, FP, FN hyp+ref combo.
                count_dict, cat_dict = evaluate_edits(hyp_dict, ref_dict, best_dict, sent_id, args)
                best_dict += Counter(count_dict)
                best_cats = merge_dict(best_cats, cat_dict)
            
            tp = best_dict['tp']
            fp = best_dict['fp']
            fn = best_dict['fn']

            p  = (1.0 * tp) / (tp + fp) if (tp + fp) > 0 else 0
            r  = (1.0 * tp) / (tp + fn)  if (tp + fn) > 0 else 0
            f_score = (1.0+BETA*BETA) * p * r / (BETA*BETA*p+r) if (p+r) > 0 else 0

            print("Write into files...")
            with file_writer.as_default():
                tf.summary.scalar('epoch_errant_precision', p, step)
                tf.summary.scalar('epoch_errant_recall', r, step)
                tf.summary.scalar('epoch_errant_f_score', f_score, step)
            print("End of writing into files...")
            

            print("Write errors...")
            with file_writer.as_default():
                text_lines = [k + ": " + str(best_cats[k]) + '\n' for k in best_cats.keys()]
                text = "".join(text_lines)
                print(text)
                tf.summary.text("errors", text, step)
            print("End of writing errors...")

        print("Write predictions...")
        with file_writer.as_default():
            text = "  \n".join(tokenized_predicted_sentences[0:40])
            print(text)
            tf.summary.text("predictions", text, step)
        with open(predictions_filepath, "w") as file:
            for sentence in tokenized_predicted_sentences:
                file.write(sentence + "\n")
        print("End of writing predictions...")

        return f_score

    while True:
        if os.path.isdir(MODEL_CHECKPOINT_PATH):
            unevaluated = [f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.startswith('ckpt')]
            numbers = np.array([int(u[5:]) for u in unevaluated])
            numbers = sorted(numbers)
            unevaluated = ["ckpt-" + str(number) for number in numbers]
            
            for unevaluated_checkpoint in unevaluated:
                try:
                    fscore_dev = generate_and_score(unevaluated_checkpoint, dev_dataset, dev_source_sentences, dev_gold_edits, OUTPUT_DIR_DEV,
                                       FILE_DEV_PREDICTIONS, dev_ref_m2, EVAL_TYPE_DEV)
                    fscore_test = generate_and_score(unevaluated_checkpoint, test_dataset, test_source_sentences, test_gold_edits, OUTPUT_DIR_TEST,
                                       FILE_TEST_PREDICTIONS, test_ref_m2, EVAL_TYPE_TEST)
                    
                    for i, dataset_zip in enumerate(datasets):
                        source_sentences, gold_edits, dataset_path = dataset_zip
                        dataset = get_dataset_pipeline(source_sentences)
                        output_dir = os.path.splitext(os.path.basename(dataset_path))[0]
                        file_predictions = os.path.splitext(os.path.basename(dataset_path))[0] + "_prediction.txt"
                        fscore = generate_and_score(unevaluated_checkpoint, dataset, source_sentences, gold_edits, output_dir, file_predictions, 
                                                    refs[i], eval_types[i])
                    
                    if BEST_CKPT_FILENAME and fscore_dev >= BEST_CKPT_FSCORE:
                        BEST_CKPT_NAME = unevaluated_checkpoint
                        BEST_CKPT_FSCORE = fscore_dev
                        
                        json_object = json.dumps({
                             "name": BEST_CKPT_NAME,
                             "fscore": BEST_CKPT_FSCORE
                        })

                        with open(BEST_CKPT_FILENAME, "w") as outfile:
                            outfile.write(json_object)
                    else:
                        print(f"Delete: {os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint)}")
                        shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint))
                        # Delete model with optimizer:
                        opt_dir = os.path.join(MODEL_CHECKPOINT_PATH, "optimizer")
                        selected_files = []
                        for f in os.listdir(opt_dir):
                            if f.startswith(unevaluated_checkpoint):
                                selected_files.append(f)

                        for selected_file in selected_files:
                            os.remove(os.path.join(opt_dir, selected_file))
                except Exception as e:
                    print(e)
                    print("Something went wrong... Try again...")

        time.sleep(10)

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

from m2scorer.levenshtein import batch_multi_pre_rec_f1
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils.udpipe_tokenizer.udpipe_tokenizer import UDPipeTokenizer


def main():
    with open('config-eval.json') as json_file:
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
    
    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']
    
    # loss
    LOSS = config['loss']
    
    # logs
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']

    # evaluation
    MAX_UNCHANGED_WORDS = config['max_unchanged_words']
    BETA = config['beta']
    IGNORE_WHITESPACE_CASING = config['ignore_whitespace_casing']
    VERBOSE = config['verbose']
    VERY_VERBOSE = config['very_verbose']
    
    tf.random.set_seed(SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    
    
    # loading of dataset:
    def split_features_and_labels(input_batch):
       features = {key: tensor for key, tensor in input_batch.items() if key in ['input_ids', 'attention_mask', 'decoder_input_ids']}
       labels = {key: tensor for key, tensor in input_batch.items() if key in ['labels']}
       if len(features) == 1:
           features = list(features.values())[0]
       if len(labels) == 1:
           labels = list(labels.values())[0]
       if isinstance(labels, dict) and len(labels) == 0:
           return features
       else:
           return features, labels
        
    def get_tokenized_sentences(line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, max_length=MAX_LENGTH, truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask']

    def create_error_line(line):
        input_ids, attention_mask = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32])
        dato = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
        }
        return dato

    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA)
        
    dataset =  tf.data.Dataset.from_tensor_slices((dev_source_sentences))
    dataset = dataset.map(create_error_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes={'input_ids': [None], 'attention_mask': [None]})
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    
    with strategy.scope():
        if OPTIMIZER_NAME == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=OPTIMIZER_PARAMS['lr'])
        elif OPTIMIZER_NAME == 'AdamW':
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=OPTIMIZER_PARAMS['lr'])
        elif OPTIMIZER_NAME == 'Adafactor':
            optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate=OPTIMIZER_PARAMS['lr'])
        elif OPTIMIZER_NAME == 'AdaptiveAdam':
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
                beta_1=beta1,
                beta_2=beta2,
                epsilon=epsilon)
    
    with strategy.scope(): 
        loss = None   
        if LOSS == "SCC":
            class MaskedSparseCategoricalCrossEntropy(tf.keras.losses.Loss):
                # source: https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/modeling_tf_utils.py#L210
                def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name=None):
                    super().__init__(reduction, name)
                    self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)

                def call(self, y_true, y_pred):
                    return self.hf_compute_loss(y_true, y_pred)

                def hf_compute_loss(self, labels, logits):
                    unmasked_loss = self.loss_func(tf.nn.relu(labels), logits)
                    loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
                    masked_loss = unmasked_loss * loss_mask
                    reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
                    return reduced_masked_loss     
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

    udpipe_tokenizer = UDPipeTokenizer("cs")

    while True:
        if os.path.isdir(MODEL_CHECKPOINT_PATH):
            unevaluated = [f for f in os.listdir(MODEL_CHECKPOINT_PATH) if f.startswith('ckpt')]
            
            for unevaluated_checkpoint in unevaluated:
                try:
                    step = int(unevaluated_checkpoint[5:])
                    result_dir = os.path.join(MODEL_CHECKPOINT_PATH, "results")

                    model.load_weights(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint + "/")).expect_partial()

                    predicted_sentences = []

                    for i, batch in enumerate(dataset):
                        print(f"Evaluating {i+1}. batch...") 
                        preds = model.generate(batch['input_ids'])
                        batch_sentences = tokenizer.batch_decode(preds, skip_special_tokens=True)
                        predicted_sentences = predicted_sentences + batch_sentences

                    tokenized_predicted_sentences = []

                    for line in predicted_sentences:
                        for sentence in udpipe_tokenizer.tokenize(line):
                            tokenized_predicted_sentences.append(" ".join([token.string for token in sentence]))

                    p, r, f1 = batch_multi_pre_rec_f1(tokenized_predicted_sentences, dev_source_sentences, dev_gold_edits, 
                                                      MAX_UNCHANGED_WORDS, BETA, IGNORE_WHITESPACE_CASING, VERBOSE, VERY_VERBOSE)

                    file_writer = tf.summary.create_file_writer(result_dir)
                    with file_writer.as_default():
                        tf.summary.scalar('epoch_precision', p, step)
                        tf.summary.scalar('epoch_recall', r, step)
                        tf.summary.scalar('epoch_f1', f1, step)

                        text = "\n".join(predicted_sentences[0:20])
                        print(text)
                        tf.summary.text("predictions", text, step)

                    print(f"Delete: {os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint)}")
                    shutil.rmtree(os.path.join(MODEL_CHECKPOINT_PATH, unevaluated_checkpoint))
                except:
                    print("Something went wrong... Try again...")

        time.sleep(10)

if __name__ == '__main__':
    main()



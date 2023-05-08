import sys
sys.path.append('..')

import os
import tensorflow as tf
import keras
import aspell

from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq 
from transformers import TFEncoderDecoderModel, BertTokenizer
from transformers import AutoTokenizer, AutoConfig

import json

from m2scorer.util import paragraphs
from m2scorer.util import smart_open
from m2scorer.levenshtein import batch_multi_pre_rec_f1
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import load_data
from utils import introduce_errors 

from multiprocessing import Process, Queue
import random
import multiprocessing

def main():
    with open('config.json') as json_file:
        config = json.load(json_file)

    tf.random.set_seed(config['seed'])

    MAX_LENGTH = config['max_length']

    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']

    STEPS_PER_EPOCH = config['steps_per_epoch']
    EPOCHS = config['epochs']
    SHUFFLE_BUFFER = config['shuffle_buffer']

    lang = config['lang']
    token_file = config['token_file']
    tokens = introduce_errors.get_token_vocabulary(token_file)
    characters = introduce_errors.get_char_vocabulary(lang)
    aspell_speller = aspell.Speller('lang', lang)
    token_err_distribution = config['token_err_distribution']
    char_err_distribution = config['char_err_distribution']
    token_err_prob = config['token_err_prob']   
    char_err_prob = config['char_err_prob']


    tokenizer = AutoTokenizer.from_pretrained("../utils/out/")

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


    queue = Queue(2 * NUM_PARALLEL)
    gel = load_data.GenereteErrorLine(tokens, characters, lang, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob)

    process = Process(target=load_data.run_proccesses_on_files, args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH,))
    process.start()

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(queue.get, None),
        output_types={
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32,
                    "labels": tf.int32,
                    "decoder_input_ids": tf.int32
                },
        output_shapes={
                    "input_ids": (None, ),
                    "attention_mask": (None, ),
                    "labels": (None, ),
                    "decoder_input_ids": (None, )
                })

    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x['input_ids'])[0],
            bucket_boundaries=[16, 32, 48, 64, 80, 96, 112],
            # bucket_batch_sizes=[128, 64, 42, 32, 25, 21, 18, 16]
            bucket_batch_sizes=[1, 1, 1, 1 , 1 , 1 , 1, 1]
    )
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # Number of batches to prefetch


    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)

    optimizer_name = config['optimizer']['name']
    optimizer_params = config['optimizer']['params']

    with strategy.scope():
        if optimizer_name == 'Adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_params['lr'])
        elif optimizer_name == 'AdamW':
            optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=optimizer_params['lr'])
        elif optimizer_name == 'Adafactor':
            optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate=optimizer_params['lr'])
        elif optimizer_name == 'AdaptiveAdam':
            class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
                def __init__(self, warmup_steps, d_model):
                    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
                    self.d_model = tf.cast(d_model, tf.float32)

                def __call__(self, step):
                    step = tf.cast(step, tf.float32)
                    lr = (1.0/tf.math.sqrt(self.d_model)) * tf.math.minimum(1.0 / tf.math.sqrt(step), (1.0 / tf.math.sqrt(self.warmup_steps)) * ((1.0 * step) / self.warmup_steps))
                    return lr

            lr = LRSchedule(optimizer_params['warmup_steps'], MAX_LENGTH)
            beta1 = optimizer_params['beta1']
            beta2 = optimizer_params['beta2']
            epsilon = optimizer_params['epsilon']
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=beta1,
                beta_2=beta2,
                epsilon=epsilon)

    with strategy.scope(): 
        loss = None   
        if config['loss'] == "SCC":
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
        from transformers import BartConfig
        config = BartConfig(
            vocab_size=32_000,
            max_position_embeddings=256,
            encoder_layers=6,
            encoder_ffn_dim=2048,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            activation_function="relu",
            d_model=512,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            init_std=0.02,
            classifier_dropout=0.0,
            scale_embedding=True,
            use_cache=True,
            num_labels=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            is_encoder_decoder=True,
            decoder_start_token_id=2,
            forced_eos_token_id=2,
        )
        model = TFAutoModelForSeq2SeqLM.from_config(config)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    print(model.optimizer)
    callbacks = []
    if STEPS_PER_EPOCH:
        model.fit(dataset, callbacks=callbacks, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    else:
        model.fit(dataset, callbacks=callbacks, epochs=EPOCHS)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
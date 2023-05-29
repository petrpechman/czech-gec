import os
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

from tensorflow.keras import mixed_precision

def main(batch_size: int, max_length: int, config: str, filename: str):
    MAX_LENGTH = max_length
    BATCH_SIZE = batch_size
    CONFIG = config

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

    tf.random.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    strategy = tf.distribute.MirroredStrategy()
    num_div = strategy.num_replicas_in_sync
    print('Number of devices: %d' % num_div)

    def get_tokenized_sentences(line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, text_target=line, max_length=max_length, padding='max_length', truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask'], tokenized['labels']

    def tokenize_line(line, max_length):
        input_ids, attention_mask, labels = tf.numpy_function(get_tokenized_sentences, inp=[line], Tout=[tf.int32, tf.int32, tf.int32])
        decoder_input_ids = tf.roll(labels, shift=1, axis=1)
        dato = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'decoder_input_ids': decoder_input_ids[0],
            'labels': labels[0]
        }
        return dato

    def ensure_shapes(input_dict, max_length):
        return {key: tf.ensure_shape(val, (max_length)) for key, val in input_dict.items()}

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

    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.map(lambda line: tokenize_line(line, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_dict: ensure_shapes(input_dict, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if USE_F16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

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

    if USE_F16:
        model.model.encoder.embed_scale = tf.cast(model.model.encoder.embed_scale, tf.float16)
        model.model.decoder.embed_scale = tf.cast(model.model.decoder.embed_scale, tf.float16)

    model.fit(dataset, epochs=2, steps_per_epoch=4)

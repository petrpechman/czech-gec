# %%
import sys
sys.path.append('..')


# %%
import os
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
import json

# from m2scorer.util import paragraphs
from m2scorer.util import smart_open
from m2scorer.levenshtein import batch_multi_pre_rec_f1
from m2scorer.m2scorer import load_annotation

from tensorflow.keras import mixed_precision

from utils import load_data
from utils import introduce_errors 

def main():
    # %%
    with open('config-eval.json') as json_file:
        config = json.load(json_file)
    
    SEED = config['seed']
    
    # data loading
    DATA_PATH = config['data_path']
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
    LOG_FILE = config['log_file']
    PROFILE_BATCH = config['profile_batch']
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']
    
    # %%
    tf.random.set_seed(SEED)
    
    # %%
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    
    
    # %%
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
        
    def get_tokenized_sentences(line, gold_line):
        line = line.decode('utf-8')
        gold_line = gold_line.decode('utf-8')
        tokenized = tokenizer(line, text_target=gold_line, padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors="tf")
        return tokenized['input_ids'], tokenized['attention_mask'], tokenized['labels']

    def create_error_line(line, gold_line):
        input_ids, attention_mask, labels = tf.numpy_function(get_tokenized_sentences, inp=[line, gold_line], Tout=[tf.int32, tf.int32, tf.int32])
        labels = labels[0]
        dato = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'labels': labels[1:],
            "decoder_input_ids": labels[:-1]
        }
        return dato

    def ensure_shapes(x):
        return {key: tf.ensure_shape(val, (MAX_LENGTH - 1)) if key in ["labels", "decoder_input_ids"] else tf.ensure_shape(val, (MAX_LENGTH)) for key, val in x.items()}
    
    print(M2_DATA)
    dev_source_sentences, dev_gold_edits = load_annotation(M2_DATA)
    gold_sentences = dev_source_sentences
        
    dataset =  tf.data.Dataset.from_tensor_slices((dev_source_sentences, gold_sentences))
    dataset = dataset.map(create_error_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(ensure_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # %%
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    
    # %%
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync)
    
    # %%
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
    
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(MODEL_CHECKPOINT_PATH, 'ckpt-{epoch}/'),
    #     save_weights_only=True)
    
    # model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=MODEL_CHECKPOINT_PATH,
    #     save_weights_only=True)
    
    # model.fit(dataset, callbacks=[model_checkpoint], epochs=1)


    # model.predict(dataset.take(1))
    model.load_weights(filepath=MODEL_CHECKPOINT_PATH)
    model.generate(["Neco tu je napsane..."])


# %%
if __name__ == '__main__':
    main()



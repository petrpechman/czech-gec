# %%
import sys
sys.path.append('../..')

# %%
import os
import tensorflow as tf
from create_errors import introduce_errors
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

# %%
with open('config.json') as json_file:
    config = json.load(json_file)

# %%
tf.random.set_seed(config['seed'])
mixed_precision.set_global_policy('mixed_float16')

# %%
USE_MODEL = config['model']
DATA_PATHS = config['data_paths']
NUM_PARALLEL = config['num_parallel']
BATCH_SIZE_PER_REPLICE = config['batch_size_per_replica']
MAX_LENGTH = config['max_length']
STEPS_PER_EPOCH = config['steps_per_epoch']
EPOCHS = config['epochs']
SHUFFLE_BUFFER = config['shuffle_buffer']

# %%
print(f"Batch size per replica: {BATCH_SIZE_PER_REPLICE}")
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)
BATCH_SIZE = BATCH_SIZE_PER_REPLICE * strategy.num_replicas_in_sync

# %%
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
            def __init__(self):
                self.loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
            
            def call(self, y_true, y_pred):
                return self.hf_compute_loss(y_true, y_pred)

            def hf_compute_loss(self, labels, logits):
                unmasked_loss = self.loss_func(tf.nn.relu(labels), logits)
                loss_mask = tf.cast(labels != -100, dtype=unmasked_loss.dtype)
                masked_loss = unmasked_loss * loss_mask
                reduced_masked_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(loss_mask)
                return reduced_masked_loss
        
        loss = MaskedSparseCategoricalCrossEntropy()
    

# %%
lang = config['lang']
token_file = config['token_file']
tokens = introduce_errors.get_token_vocabulary(token_file)
characters = introduce_errors.get_char_vocabulary(lang)
aspell_speller = aspell.Speller('lang', lang)
token_err_distribution = config['token_err_distribution']
char_err_distribution = config['char_err_distribution']
token_err_prob = config['token_err_prob']   
char_err_prob = config['char_err_prob']

# %%
tokenizer = AutoTokenizer.from_pretrained(config['model'])

# %%
class GenereteErrorLine():

    def __init__(self, tokens, characters, aspell_speller, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob, token_std_dev=0.2, char_std_dev=0.01):
        self.tokens = tokens
        self.characters = characters
        self.aspell_speller = aspell_speller
        self.token_err_distribution = token_err_distribution
        self.char_err_distribution = char_err_distribution
        self.token_err_prob = token_err_prob
        self.token_std_dev = token_std_dev
        self.char_err_prob = char_err_prob
        self.char_std_dev = char_std_dev

    def __call__(self, line):
        line = line.decode('utf-8')
        token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = self.token_err_distribution
        char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob = self.char_err_distribution
        line = line.strip('\n')
        
        # introduce word-level errors
        line = introduce_errors.introduce_token_level_errors_on_sentence(line.split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, float(self.token_err_prob), float(self.token_std_dev),
                                                        self.tokens, self.aspell_speller)
        if '\t' in line or '\n' in line:
            raise ValueError('!!! Error !!! ' + line)
        # introduce spelling errors
        line = introduce_errors.introduce_char_level_errors_on_sentence(line, char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob,
                                                       change_diacritics_prob, float(self.char_err_prob), float(self.char_std_dev),
                                                       self.characters)
        return line
    
gel = GenereteErrorLine(tokens, characters, aspell_speller, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob)

# %%
def get_tokenized_sentences(error_line, label_line):
    error_line = error_line.decode('utf-8')
    label_line = label_line.decode('utf-8')
    tokenized = tokenizer(error_line, text_target=label_line, max_length=MAX_LENGTH, padding='max_length', truncation=True, return_tensors="tf")
    return tokenized['input_ids'], tokenized['attention_mask'], tokenized['labels']

def create_error_line(line):
    error_line = tf.numpy_function(gel, inp=[line], Tout=[tf.string])[0]
    label_line = line
    input_ids, attention_mask, labels = tf.numpy_function(get_tokenized_sentences, inp=[error_line, label_line], Tout=[tf.int32, tf.int32, tf.int32])
    decoder_input_ids = tf.roll(labels, shift=1, axis=1)
    dato = {
        'input_ids': input_ids[0],
        'attention_mask': attention_mask[0],
        'decoder_input_ids': decoder_input_ids[0],
        'labels': labels[0]
    }
    return dato

def ensure_shapes(input_dict):
    return {key: tf.ensure_shape(val, (MAX_LENGTH)) for key, val in input_dict.items()}

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


# %%
dataset = tf.data.TextLineDataset(DATA_PATHS, num_parallel_reads=NUM_PARALLEL)

dataset = dataset.map(create_error_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(ensure_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.ignore_errors()
dataset = dataset.shuffle(SHUFFLE_BUFFER)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# %%
with strategy.scope():
    if config["pretrained"]:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(config['model'])
    else:
        config = AutoConfig.from_pretrained(config['model'])
        model = TFAutoModelForSeq2SeqLM.from_config(config)
    
    if loss:
        model.compile(optimizer=optimizer, loss=loss)
    else:
        model.compile(optimizer=optimizer)

# %% [markdown]
# ---
# ### Evaluation

# %%
max_unchanged_words=2
beta = 0.5
ignore_whitespace_casing= False
verbose = False
very_verbose = False

dev_input = config['evaluation_input']
dev_gold = config['evaluation_gold']

# load source sentences and gold edits
fin = smart_open(dev_input, 'r')
dev_input_sentences = [line.strip() for line in fin.readlines()]
fin.close()

dev_source_sentences, dev_gold_edits = load_annotation(dev_gold)

# %%
class Evaluation(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, nth, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose, 
                 dev_input_sentences, dev_source_sentences, dev_gold_edits, ensure_shapes, split_features_and_labels, batch_size):
        self.tokenzer = tokenizer
        self.nth = nth
        self.max_unchanged_words = max_unchanged_words
        self.beta = beta
        self.ignore_whitespace_casing = ignore_whitespace_casing
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.dev_input_sentences = dev_input_sentences
        self.dev_source_sentences = dev_source_sentences
        self.dev_gold_edits = dev_gold_edits

        self.ensure_shapes = ensure_shapes
        self.split_features_and_labels = split_features_and_labels
        self.batch_size = batch_size

    def get_tokenized_sentence(self, line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, padding='max_length', truncation=True, return_tensors="tf")
        return tokenized['input_ids']

    def create_tokenized_line(self, line):
        input_ids = tf.numpy_function(self.get_tokenized_sentence, inp=[line], Tout=tf.int32)
        dato = {
            'input_ids': input_ids[0]
        }
        return dato

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.nth == 0:
            try:
                predicted_sentences = []
                val_dataset = tf.data.Dataset.from_tensor_slices(self.dev_input_sentences)
                val_dataset = val_dataset.map(self.create_tokenized_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                val_dataset = val_dataset.map(self.ensure_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                val_dataset = val_dataset.map(self.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                val_dataset = val_dataset.batch(self.batch_size)

                for batch in val_dataset: 
                    outs = model.generate(batch)
                    for out in outs:
                        predicted_sentence = tokenizer.decode(out)
                        predicted_sentences.append(predicted_sentence)
                
                p, r, f1 = batch_multi_pre_rec_f1(predicted_sentences, self.dev_source_sentences, self.dev_gold_edits, 
                                                  self.max_unchanged_words, self.beta, self.ignore_whitespace_casing, self.verbose, self.very_verbose)
                print("Precision   : %.4f" % p)
                print("Recall      : %.4f" % r)
                print("F_%.1f       : %.4f" % (self.beta, f1))
            except:
                print("No predictions...")

callbacks = [
    Evaluation(tokenizer=tokenizer, nth=config['evaluation_every_nth'],
               max_unchanged_words=max_unchanged_words, beta=beta, ignore_whitespace_casing=ignore_whitespace_casing,
               verbose=verbose, very_verbose=very_verbose, dev_input_sentences=dev_input_sentences, dev_source_sentences=dev_source_sentences,
               dev_gold_edits=dev_gold_edits, ensure_shapes=ensure_shapes, split_features_and_labels=split_features_and_labels,
               batch_size=config['batch_size_eval']),
    tf.keras.callbacks.TensorBoard(log_dir=config['log_file'], profile_batch=config['profile_batch']),
    tf.keras.callbacks.ModelCheckpoint(filepath=config['model_checkpoint_path'], save_weights_only=True, save_freq='epoch')
]

# %% [markdown]
# ---

# %% [markdown]
# ### Train

# %%
if STEPS_PER_EPOCH:
    model.fit(dataset, callbacks=callbacks, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
else:
    model.fit(dataset, callbacks=callbacks, epochs=EPOCHS)

# %% [markdown]
# ---

# %%
# checkpoint_filepath = './tmp/checkpoint/' # must be folder (/ at the end)

# model.load_weights(checkpoint_filepath)

# %% [markdown]
# ---



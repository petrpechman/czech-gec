# %%
import sys
sys.path.append('..')

# %%
EVALUATOR = False
if len(sys.argv) > 1:
    if sys.argv[1] == "-e":
        EVALUATOR = True

# %%
import os
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import json

# from m2scorer.util import paragraphs
# from m2scorer.util import smart_open
# from m2scorer.levenshtein import batch_multi_pre_rec_f1
# from m2scorer.m2scorer import load_annotation

# from tensorflow.keras import mixed_precision

from utils import load_data
from utils import introduce_errors 

from multiprocessing import Process, Queue, Manager
import multiprocessing

def main():
    # %%
    with open('config.json') as json_file:
        config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    MAX_LENGTH = config['max_length']
    SHUFFLE_BUFFER = config['shuffle_buffer']

    # model
    MODEL = config['model']
    PRETRAINED = config['pretrained']
    STEPS_PER_EPOCH = config['steps_per_epoch']
    EPOCHS = config['epochs']

    # optimizer
    OPTIMIZER_NAME = config['optimizer']['name']
    OPTIMIZER_PARAMS = config['optimizer']['params']

    # loss
    LOSS = config['loss']

    # GEL config
    LANG = config['lang']
    TOKEN_FILE = config['token_file']
    TOKEN_ERR_DISTRIBUTION = config['token_err_distribution']
    CHAR_ERR_DISTRIBUTION = config['char_err_distribution']
    TOKEN_ERR_PROB = config['token_err_prob']   
    CHAR_ERR_PROB = config['char_err_prob']

    # logs
    LOG_FILE = config['log_file']
    PROFILE_BATCH = config['profile_batch']
    MODEL_CHECKPOINT_PATH = config['model_checkpoint_path']

    # %%
    tf.random.set_seed(config['seed'])

    # %%
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # %%
    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)
    # aspell_speller = aspell.Speller('lang', LANG)

    # %%
    # new loading of dataset:

    # multiprocessing.set_start_method('spawn')   

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

    manager = Manager()
    queue = manager.Queue(2 * NUM_PARALLEL)
    gel = load_data.GenereteErrorLine(
            tokens, characters, LANG, 
            TOKEN_ERR_DISTRIBUTION, CHAR_ERR_DISTRIBUTION, 
            TOKEN_ERR_PROB, CHAR_ERR_PROB)

    process = Process(
                target=load_data.data_generator, 
                args=(queue, DATA_PATHS, NUM_PARALLEL, gel, tokenizer, MAX_LENGTH,))

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
    dataset = dataset.shuffle(SHUFFLE_BUFFER)
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x['input_ids'])[0],
            # bucket_boundaries=[16, 32, 48, 64, 80, 96, 112],
            # bucket_batch_sizes=[1, 1, 1, 1 , 1 , 1 , 1, 1]
            bucket_boundaries=[32, 64, 96],
            bucket_batch_sizes=[48, 20, 12, 8]
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # Number of batches to prefetch

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


    # %%
    # tokenizer_eval = AutoTokenizer.from_pretrained(...)

    # %%
    with strategy.scope():
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    # %%
    print(model.optimizer)

    # %% [markdown]
    # ---
    # ### Evaluation

    # %%
    # max_unchanged_words=2
    # beta = 0.5
    # ignore_whitespace_casing= False
    # verbose = False
    # very_verbose = False

    # dev_input = config['evaluation_input']
    # dev_gold = config['evaluation_gold']

    # # load source sentences and gold edits
    # fin = smart_open(dev_input, 'r')
    # dev_input_sentences = [line.strip() for line in fin.readlines()]
    # fin.close()

    # dev_source_sentences, dev_gold_edits = load_annotation(dev_gold)

    # %%
    # class Evaluation(tf.keras.callbacks.Callback):
    #     def __init__(self, tokenizer, max_length, nth, max_unchanged_words, beta, ignore_whitespace_casing, verbose, very_verbose, 
    #                  dev_input_sentences, dev_source_sentences, dev_gold_edits, ensure_shapes, split_features_and_labels, batch_size):
    #         self.tokenizer = tokenizer
    #         self.max_length = max_length
    #         self.nth = nth
    #         self.max_unchanged_words = max_unchanged_words
    #         self.beta = beta
    #         self.ignore_whitespace_casing = ignore_whitespace_casing
    #         self.verbose = verbose
    #         self.very_verbose = very_verbose
    #         self.dev_input_sentences = dev_input_sentences
    #         self.dev_source_sentences = dev_source_sentences
    #         self.dev_gold_edits = dev_gold_edits

    #         self.ensure_shapes = ensure_shapes
    #         self.split_features_and_labels = split_features_and_labels
    #         self.batch_size = batch_size

    #     def get_tokenized_sentence(self, line):
    #         line = line.decode('utf-8')
    #         tokenized = self.tokenizer(line, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="tf")
    #         return tokenized['input_ids']

    #     def create_tokenized_line(self, line):
    #         input_ids = tf.numpy_function(self.get_tokenized_sentence, inp=[line], Tout=tf.int32)
    #         dato = {
    #             'input_ids': input_ids[0]
    #         }
    #         return dato

    #     def on_epoch_end(self, epoch, logs=None):
    #         if epoch % self.nth == 0:
    #             try:
    #                 predicted_sentences = []
    #                 val_dataset = tf.data.Dataset.from_tensor_slices(self.dev_input_sentences)
    #                 val_dataset = val_dataset.map(self.create_tokenized_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #                 val_dataset = val_dataset.map(self.ensure_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #                 val_dataset = val_dataset.map(self.split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #                 val_dataset = val_dataset.batch(self.batch_size)

    #                 for batch in val_dataset: 
    #                     outs = model.generate(batch)
    #                     for out in outs:
    #                         predicted_sentence = tokenizer.decode(out)
    #                         predicted_sentences.append(predicted_sentence)

    #                 p, r, f1 = batch_multi_pre_rec_f1(predicted_sentences, self.dev_source_sentences, self.dev_gold_edits, 
    #                                                   self.max_unchanged_words, self.beta, self.ignore_whitespace_casing, self.verbose, self.very_verbose)
    #                 print("Precision   : %.4f" % p)
    #                 print("Recall      : %.4f" % r)
    #                 print("F_%.1f       : %.4f" % (self.beta, f1))
    #             except:
    #                 print("No predictions...")

    # %%

    # callbacks = [
    #     Evaluation(tokenizer=tokenizer_eval, max_length=MAX_LENGTH ,nth=config['evaluation_every_nth'],
    #                max_unchanged_words=max_unchanged_words, beta=beta, ignore_whitespace_casing=ignore_whitespace_casing,
    #                verbose=verbose, very_verbose=very_verbose, dev_input_sentences=dev_input_sentences, dev_source_sentences=dev_source_sentences,
    #                dev_gold_edits=dev_gold_edits, ensure_shapes=ensure_shapes, split_features_and_labels=split_features_and_labels,
    #                batch_size=config['batch_size_eval']),
    #     tf.keras.callbacks.TensorBoard(log_dir=config['log_file'], profile_batch=config['profile_batch']),
    #     tf.keras.callbacks.ModelCheckpoint(filepath=config['model_checkpoint_path'], save_weights_only=True, save_freq='epoch')
    # ]

    # %%
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_CHECKPOINT_PATH, 'ckpt-{epoch}'),
        save_weights_only=True)

    # %%
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_FILE, profile_batch=PROFILE_BATCH),
    ]

    # %% [markdown]
    # ---

    # %% [markdown]
    # ### Train

    # %%
    if EVALUATOR:
        sidecar = tf.keras.utils.SidecarEvaluator(
            model,
            dataset,
            checkpoint_dir=MODEL_CHECKPOINT_PATH,
            steps=300,
            max_evaluations=None,
            callbacks=[]
        )
        sidecar.start()
    else:

        if STEPS_PER_EPOCH:
            model.fit(dataset, callbacks=[model_checkpoint], epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
        else:
            model.fit(dataset, callbacks=[model_checkpoint], epochs=EPOCHS)

    # %% [markdown]
    # ---

    # %%
    # checkpoint_filepath = './tmp/checkpoint/' # must be folder (/ at the end)

    # model.load_weights(checkpoint_filepath)

    # %% [markdown]
    # ---

# %%
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()



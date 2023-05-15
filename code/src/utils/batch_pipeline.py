import sys
sys.path.append('..')

import os
import tensorflow as tf

from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import json

from utils import load_data
from utils import introduce_errors 

from multiprocessing import Process, Queue, Manager
import multiprocessing

LINE = "Monkey D. Luffy, also known as Straw Hat Luffy and commonly as Straw Hat,[10] is the founder and captain of the increasingly" \
        " infamous and powerful Straw Hat Pirates, as well as the most powerful of its top fighters.[26][27] He desires to find" \
        " the legendary treasure left behind by the late Gol D. Roger and thereby become the Pirate King,[28] which would help" \
        " facilitate an unknown dream of his that he has told only to Shanks, his brothers, and crew.[29][30] He believes that" \
        " being the Pirate King means having the most freedom in the world.[31] Born in Foosha Village, Luffy is the son of" \
        " Monkey D. Dragon, the leader of the Revolutionary Army,[32] and the grandson of the Marine hero Monkey D. Garp,[33]" \
        " and their family carries the initial and Will of D. At age 7, Luffy accidentally ate the Gomu Gomu no Mi, which turned his body into rubber." \
        " [34] Shanks also gave Luffy the very straw hat that has become Luffy's signature accessory, having gifted it to the boy as part of a promise" \
        " for them to meet again someday after he became a great pirate.[35] Growing up on Dawn Island under the care of Curly Dadan,[36] Luffy befriended" \
        " and became sworn brothers of the late Fire Fist Portgas D. Ace[37] and Revolutionary Chief-of-Staff Sabo.[38] Luffy has gone up against numerous" \
        " global powers around him, starting with fighting the most powerful pirates in the East Blue and moving to clashes against the Marines, " \
        "Seven Warlords of the Sea, Cipher Pol, World Nobles, and even the Four Emperors of the Grand Line, emerging victorious in a majority of" \
        " these battles. He invaded and indirectly caused the annihilation of Enies Lobby, escaped the impregnable Impel Down, and was a focal figure " \
        "in the Summit War of Marineford. He has also either defeated or befriended seven of the eleven known past or present Warlords prior to the" \
        " organization's dissolution. Furthermore, Luffy has invaded the territory of the Four Emperors on multiple occasions, and eventually managed to" \
        "defeat one. Luffy's accomplishments and heritage have caused him to be labeled as a Dangerous Future Element while in the process gaining a " \
        "reputation for being reckless and, in some cases, insane, earning the wrath of Fleet Admiral Sakazuki, the Marine Headquarters, and even " \
        "the World Government.[39] Luffy also has a penchant for attracting followers and has unwillingly been named the leader of the Straw Hat Grand Fleet" \
        ", consisting of seven pirate crews who swore to come to his aid whenever he wishes. After learning of this and his exploits against the Big Mom Pirates," \
        " the press labeled him the Fifth Emperor of the Sea, though many prominent figures initially considered this to be exaggerated.[40] However, after" \
        " defeating Kaidou during the Raid on Onigashima, Luffy was officially declared as one of the Four Emperors by the World Government along with Buggy," \
        " replacing Kaidou and Big Mom.[3] Luffy has made tremendous strides in his life of piracy, with his bounty heavily reflecting this fact. He gained " \
        "his first bounty of Beli30,000,000 for defeating the strongest pirate captains of the East Blue, which then increased to Beli100,000,000 after defeating" \
        " Crocodile in Arabasta. After his crew's invasion into and escape from Enies Lobby, his bounty was increased to Beli300,000,000. His sizeable bounty upon" \
        " arriving at the Sabaody Archipelago caused Luffy, along with Zoro to be included among the eleven Super Rookies, pirates who simultaneously reached the" \
        "Red Line with bounties of over Beli100,000,000 shortly before the Summit War.[41] He, the other ten Super Rookies, and Marshall D. Teach would go on to" \
        " be referred to as the Worst Generation.[42] Two years after the war, with his bounty increased to Beli400,000,000, he entered the New World and began " \
        " challenging the Emperors and their allies directly, with his bounty going up to Beli500,000,000 after the Dressrosa Arc, and later all the way to Beli1," \
        " 500,000,000 after the global revelation that he is Sabo's brother and the existence of Straw Hat Grand Fleet becoming public, as well as the events of " \
        " the Whole Cake Island Arc. After leading the Raid on Onigashima and defeating Kaidou as well he became a member of the Four Emperors, his bounty was " \
        " increased to Beli3,000,000,000. He is the main protagonist of the manga and anime, One Piece."


def main(batch_size: int, max_length: int, config: str, num_lines: int):
    lines = [LINE] *  num_lines
    MAX_LENGTH = max_length
    BATCH_SIZE = batch_size
    CONFIG = config

    with open(CONFIG) as json_file:
        config = json.load(json_file)

    SEED = config['seed']

    # data loading
    DATA_PATHS = config['data_paths']
    NUM_PARALLEL = config['num_parallel']
    SHUFFLE_BUFFER = config['shuffle_buffer']

    # model
    MODEL = config['model']
    TOKENIZER = config['tokenizer']
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
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # %%
    tokens = introduce_errors.get_token_vocabulary(TOKEN_FILE)
    characters = introduce_errors.get_char_vocabulary(LANG)

    # %%
    # new loading of dataset:
    def tokenize_line(line, tokenizer, max_length):
        def get_tokenized_sentences(line):
            line = line.decode('utf-8')
            tokenized = tokenizer(line, text_target=line, max_length=max_length, padding='max_length', truncation=True, return_tensors="tf")
            return tokenized['input_ids'], tokenized['attention_mask'], tokenized['labels']
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

    dataset = tf.data.Dataset.from_tensor_slices((lines))
    dataset = dataset.map(lambda line: tokenize_line(line, tokenizer, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_dict: ensure_shapes(input_dict, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
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
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

        if loss:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            model.compile(optimizer=optimizer)

    model.fit(dataset, epochs=2, steps_per_epoch=4)
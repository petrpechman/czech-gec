import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from multiprocessing import Process
from tensorflow.python.client import device_lib
from tensorflow.keras import mixed_precision
import os
from transformers import BartConfig
import time

LINE = "Monkey D. Luffy, also known as Straw Hat Luffy and commonly as Straw Hat,[10] is the founder and captain of the increasingly" \
        " infamous and powerful Straw Hat Pirates, as well as the most powerful of its top fighters.[26][27] He desires to find" \
        " the legendary treasure left behind by the late Gol D. Roger and thereby become the Pirate King,[28] which would help" \
        " facilitate an unknown dream of his that he has told only to Shanks, his brothers, and crew.[29][30] He believes that" \
        " being the Pirate King means having the most freedom in the world.[31] Born in Foosha Village, Luffy is the son of"\  
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
        " increased to Beli3,000,000,000. He is the main protagonist of the manga and anime, One Piece.""


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
    
def try_batch_size(model, tokenizer, lines, batch_size, max_length, lr=0.00001) -> bool:
    print(device_lib.list_local_devices())

    dataset = tf.data.Dataset.from_tensor_slices((lines))
    dataset = dataset.map(lambda line: tokenize_line(line, tokenizer, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_dict: ensure_shapes(input_dict, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)
    model.fit(dataset, epochs=2, steps_per_epoch=2)


def get_batch_size(max_length, filename) -> int:
    NUM_LINES = 128
    MAX_BATCH_SIZE = 2049
    STEP_BATCH = 4

    lines = [LINE] *  NUM_LINES

    for batch_size in range(STEP_BATCH, MAX_BATCH_SIZE, STEP_BATCH):
        # # model_name = "facebook/bart-base"
        # # model_name = "google/mt5-small"
        # model = TFAutoModelForSeq2SeqLM.from_pretrained("./model/")
        # tokenizer = AutoTokenizer.from_pretrained("./out/")
        # policy = mixed_precision.Policy('mixed_float16')    
        # mixed_precision.set_global_policy(policy)
        
        # # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # # model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
        # try_batch_size(model, tokenizer, lines, batch_size, max_length)
        # log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
        # print(f"Allowed batch size {batch_size} for max_length {max_length}.")

        try:
            # policy = mixed_precision.Policy('mixed_float16')    
            # mixed_precision.set_global_policy(policy)

            # model_name = "facebook/bart-base"
            model_name = "google/mt5-small"

            # model = TFAutoModelForSeq2SeqLM.from_pretrained("./model/")
            # tokenizer = AutoTokenizer.from_pretrained("./out/")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            try_batch_size(model, tokenizer, lines, batch_size, max_length)
            log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
            print(f"Allowed batch size {batch_size} for max_length {max_length}.")
        except:
            return batch_size - STEP_BATCH
    return 0
        
def all_batch_sizes(filename: str):
    MAX_LENGTH = 16384
    STEP_LENGTH = 16

    batch_sizes = []
    for max_length in range(STEP_LENGTH, MAX_LENGTH, STEP_LENGTH):
        batch_size = get_batch_size(max_length, filename)
        print(f"BATCH SIZE: {batch_size}   MAX LENGHT: {max_length}")
        if batch_size == 0:
            break
        batch_sizes.append((max_length, batch_size))
    return batch_sizes

def log_data(filename: str, text: str):
    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w'
    with open(filename, append_write, encoding="utf-8") as log_file:
        print(text, file=log_file)

def main():
    filename = "mt5-small-batches.txt"

    batch_size = get_batch_size(32 ,filename)
    batch_size = get_batch_size(64 ,filename)
    batch_size = get_batch_size(96 ,filename)
    batch_size = get_batch_size(128 ,filename)

if __name__ == "__main__":
    main()

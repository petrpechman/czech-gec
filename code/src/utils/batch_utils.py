import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from multiprocessing import Process
from tensorflow.python.client import device_lib
from tensorflow.keras import mixed_precision
import os

LINE = "Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena"


def tokenize_line(line, tokenizer, max_length):
    def get_tokenized_sentences(line):
        line = line.decode('utf-8')
        tokenized = tokenizer(line, text_target=line, max_length=max_length, truncation=True, return_tensors="tf")
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
        try:
            model_name = "facebook/bart-base"
            # model_name = "google/mt5-small"

            policy = mixed_precision.Policy('mixed_float16')    
            mixed_precision.set_global_policy(policy)
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
            try_batch_size(model, tokenizer, lines, batch_size, max_length)
            log_data(filename, f"Allowed batch size {batch_size} for max_length {max_length}.")
            print(f"Allowed batch size {batch_size} for max_length {max_length}.")
        except:
            return batch_size - STEP_BATCH
        
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
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)

    # tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    # model = AutoModel.from_pretrained("facebook/bart-base")

    filename = "bart-base-batches.txt"

    batch_sizes = get_batch_size(64 ,filename)
    batch_sizes = get_batch_size(128 ,filename)
    print("BATCH:", batch_sizes)

if __name__ == "__main__":
    main()

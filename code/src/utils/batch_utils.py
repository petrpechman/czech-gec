import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

LINE = "Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena \
Nebo nevím nějaké specifiské běloruské národní tradice, protože vyrostl jsem ve městě, kde oslává Vánoc neni tak rozšiřena"


tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")


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
    
def try_batch_size(model, dataset, batch_size) -> bool:
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    try:
        model.fit(dataset, epochs=2, steps_per_epoch=2)
        return True
    except:
        return False


def get_batch_size(model, tokenizer, max_length) -> int:
    lr = 0.00001
    NUM_LINES = 128
    MAX_BATCH_SIZE = 2049
    step = 16

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)

    lines = [LINE] *  NUM_LINES
    dataset = tf.data.Dataset.from_tensor_slices((lines))
    # dataset = dataset.map(tokenize_line, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda line: tokenize_line(line, tokenizer, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda input_dict: ensure_shapes(input_dict, max_length), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for batch_size in range(step, MAX_BATCH_SIZE, step):
        if not try_batch_size(model, dataset, batch_size):
            return batch_size - step
        
def main():

    max_length = 128

    batch_size = get_batch_size(model, tokenizer, max_length)
    print(batch_size)

main()
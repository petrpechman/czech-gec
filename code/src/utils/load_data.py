from multiprocessing import Process, Queue
import tensorflow as tf
from typing import List
import random
# from . import introduce_errors
import introduce_errors
import aspell
from transformers import AutoTokenizer
import multiprocessing

class GenereteErrorLine():

    def __init__(self, tokens, characters, lang, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob, token_std_dev=0.2, char_std_dev=0.01):
        self.tokens = tokens
        self.characters = characters
        self.lang = lang
        self.token_err_distribution = token_err_distribution
        self.char_err_distribution = char_err_distribution
        self.token_err_prob = token_err_prob
        self.token_std_dev = token_std_dev
        self.char_err_prob = char_err_prob
        self.char_std_dev = char_std_dev

    def __call__(self, line, aspell_speller):
        token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = self.token_err_distribution
        char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob = self.char_err_distribution
        line = line.strip('\n')
        
        # introduce word-level errors
        line = introduce_errors.introduce_token_level_errors_on_sentence(line.split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, float(self.token_err_prob), float(self.token_std_dev),
                                                        self.tokens, aspell_speller)
        if '\t' in line or '\n' in line:
            raise ValueError('!!! Error !!! ' + line)
        # introduce spelling errors
        line = introduce_errors.introduce_char_level_errors_on_sentence(line, char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob,
                                                       change_diacritics_prob, float(self.char_err_prob), float(self.char_std_dev),
                                                       self.characters)
        return line
    

def data_generator(filename, queue, start_position, end_position, gel: GenereteErrorLine, tokenizer, max_length):
    counter = 0
    aspell_speller = aspell.Speller('lang', gel.lang)
    with open(filename, 'r') as f:
        while counter != start_position:
            f.readline()
            counter += 1

        while counter != end_position:
            line = f.readline()
            error_line = gel(line, aspell_speller)
            tokenized = tokenizer(error_line, text_target=line, max_length=max_length, truncation=True, return_tensors="tf")

            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]
            labels = tokenized['labels'][0]
            decoder_input_ids = tf.roll(labels, shift=1, axis=0)

            dato = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "decoder_input_ids": decoder_input_ids
            }
            
            queue.put(dato)

            counter += 1

            if not line: # EOF
                f.seek(0) 
                counter = 0

        

def run_processes(queue: Queue, num_parallel: int, filename: str, file_size: int, gel: GenereteErrorLine, tokenizer, max_length):
    start = random.randint(0, file_size-1)
    process_size = file_size // num_parallel
    
    positions = []
    current = start
    for i in range(num_parallel):
        positions.append(current)
        current = current + process_size
    positions.append(start)

    processes = []
    for i in range(num_parallel):
        process = Process(target=data_generator, args=(filename, queue, positions[i], positions[i+1], gel, tokenizer, max_length,))
        process.start()
        processes.append(process)

    return processes


def run_proccesses_on_files(queue: Queue, files: List[str], num_parallel: int, gel: GenereteErrorLine, tokenizer, max_length):
    index = 0
    while True:
        file = files[index]

        with open(file, 'r') as f:
            for count, _ in enumerate(f):
                pass
        file_size = count + 1
        
        processes = run_processes(queue, num_parallel, file, file_size, gel, tokenizer, max_length)
        for process in processes:
            process.join()

        index += 1
        if index == len(files):
            index = 0


# ########################################################################

def main():
    multiprocessing.set_start_method('spawn')   

    PATH = "/home/petr/Plocha/DP/czech-gec/code/data/geccc/train/sentence.input"
    NUM_PARALLEL = 3
    MAX_LENGTH = 128    


    # tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base") 

    lang = "cs"
    token_file = "/home/petr/Plocha/DP/czech-gec/code/data/vocabluraries/vocabulary_cs.tsv"
    tokens = introduce_errors.get_token_vocabulary(token_file)
    characters = introduce_errors.get_char_vocabulary(lang)
    # aspell_speller = aspell.Speller('lang', lang)
    token_err_distribution = [0.7, 0.1, 0.1, 0.1, 0]
    char_err_distribution = [0.25, 0.25, 0.25, 0.25, 0]
    token_err_prob = 0.15
    char_err_prob = 0.02    

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

    random.seed(42) 

    queue = Queue(2 * NUM_PARALLEL)
    gel = GenereteErrorLine(tokens, characters, lang, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob) 

    process = Process(target=run_proccesses_on_files, args=(queue, [PATH], NUM_PARALLEL, gel, tokenizer, MAX_LENGTH,))
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

    # dataset = dataset.map(ensure_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(split_features_and_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda x, y: tf.shape(x['input_ids'])[0],
            bucket_boundaries=[16, 32, 48, 64, 80, 96, 112],
            # bucket_batch_sizes=[128, 64, 42, 32, 25, 21, 18, 16]
            bucket_batch_sizes=[1, 1, 1, 1 , 1 , 1 , 1, 1]
    )
    dataset = dataset.prefetch(2)

    # def shuffle_batch():
    #     buffer_size = 100
    #     buffer = [] 

    #     max_batch_size = -1
    #     batch_inputs = []
    #     batch_attention_masks = []
    #     batch_labels = []
    #     batch_size_inputs = 0
    #     batch_size_labels = 0

    #     while True:
    #         try:
    #             dato = queue.get()

    #             if len(buffer) < buffer_size: 
    #                 buffer.append(dato)
    #             else:
    #                 index = random.randint(0, buffer_size - 1)
    #                 input_ids, attention_mask, labels, decoder_input_ids = buffer[index]
    #                 buffer[index] = dato    

    #                 if max_batch_size == -1:
    #                     yield (input_ids, 
    #                            attention_mask, 
    #                            labels,
    #                            decoder_input_ids)
    #                 else:
    #                     if (batch_size_inputs + len(input_ids) > max_batch_size) or (batch_size_labels + len(labels) > max_batch_size):
    #                         yield (tf.ragged.stack(batch_inputs), 
    #                                tf.ragged.stack(batch_attention_masks), 
    #                                tf.ragged.stack(batch_labels))
    #                         batch_inputs = []
    #                         batch_attention_masks = []
    #                         batch_labels = []
    #                         batch_size_inputs = 0
    #                         batch_size_labels = 0   

    #                     batch_size_inputs += len(input_ids)
    #                     batch_size_labels += len(labels)
    #                     batch_inputs.append(input_ids)
    #                     batch_attention_masks.append(attention_mask)
    #                     batch_labels.append(labels) 

    #         except queue.Empty:
    #             pass    


    # dataset = tf.data.Dataset.from_generator(
    #     # lambda: iter(queue.get, None),
    #     shuffle_batch,
    #     # output_signature=(
    #     #     tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
    #     #     tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
    #     #     tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32)
    #     #     )
    #     output_types=(tf.int32, tf.int32, tf.int32, tf.int32),
    #     output_shapes=((None, ),(None, ), (None, ), (None, ))
    #     )   


    # # dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=4))
    # # dataset = dataset.ragged_batch(4)
    # dataset = dataset.bucket_by_sequence_length(
    #         element_length_func=lambda x, y, z, w: tf.shape(x)[0],
    #         bucket_boundaries=[16, 32, 48, 64, 80, 96, 112],
    #         bucket_batch_sizes=[128, 64, 42, 32, 25, 21, 18, 16]
    # )
    # dataset = dataset.prefetch(2) # Number of batches to prefetch   

    for d in dataset:
        print(d)    

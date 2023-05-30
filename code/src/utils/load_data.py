from multiprocessing import Process, Queue
import tensorflow as tf
from typing import List
import random
from . import introduce_errors
# import introduce_errors
import aspell
from transformers import AutoTokenizer
import multiprocessing
from tokenizers import Tokenizer
from multiprocessing import Pool
from multiprocessing import Manager

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
    

def data_loader(filename, queue, start_position, end_position, gel: GenereteErrorLine, tokenizer, max_length):
    counter = 0
    aspell_speller = aspell.Speller('lang', gel.lang)
    with open(filename, 'r') as f:
        # find start position
        while counter != start_position:
            f.readline()
            counter += 1

        # read until end position
        while counter != end_position:
            line = f.readline()
            error_line = gel(line, aspell_speller)
            tokenized = tokenizer(error_line, text_target=line, max_length=max_length, truncation=True, return_tensors="np")

            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]
            tokenized_target_line = tokenized['labels'][0]

            dato = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": tokenized_target_line[1:],
                "decoder_input_ids": tokenized_target_line[:-1]
            }
            
            queue.put(dato)

            counter += 1

            if not line: # EOF
                f.seek(0) 
                counter = 0



def process_file_in_chunks(
        queue: Queue, pool: Pool, num_parallel: int, filename: str, file_size: int, 
        gel: GenereteErrorLine, tokenizer, max_length):
    
    start = random.randint(0, file_size-1)
    process_size = file_size // num_parallel

    # create list of arguments
    arguments = []

    current = start
    start_position = current
    for i in range(num_parallel):
        current = (current + process_size) % file_size
        end_position = current
        arguments.append((filename, queue, start_position, end_position, gel, tokenizer, max_length,))
        start_position = current
    end_position = start
    arguments.append((filename, queue, start_position, end_position, gel, tokenizer, max_length,))

    # start processes and wait until they finished
    pool.starmap(data_loader, arguments)


def data_generator(queue: Queue, files: List[str], num_parallel: int, gel: GenereteErrorLine, tokenizer, max_length):
    index = 0
    pool = Pool(num_parallel)

    while True:
        file = files[index]

        # get file size
        with open(file, 'r') as f:
            for count, _ in enumerate(f):
                pass
        file_size = count + 1
        
        process_file_in_chunks(queue, pool, num_parallel, file, file_size, gel, tokenizer, max_length)

        index += 1
        if index == len(files):
            index = 0



# Main is just for try:

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

    manager = Manager()
    queue = manager.Queue(2 * NUM_PARALLEL)

    gel = GenereteErrorLine(tokens, characters, lang, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob) 

    process = Process(target=data_generator, args=(queue, [PATH], NUM_PARALLEL, gel, tokenizer, MAX_LENGTH,))
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
            bucket_batch_sizes=[1, 1, 1, 1 , 1 , 1 , 1, 1]
    )
    dataset = dataset.prefetch(2)

    for d in dataset:
        print(d)    

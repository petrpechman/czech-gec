from multiprocessing import Process, Queue
import tensorflow as tf
from typing import List
import random
import introduce_errors
import aspell
from transformers import AutoTokenizer
import keras_nlp

PATH = "/home/petr/Plocha/DP/czech-gec/code/data/geccc/train/sentence.input"
NUM_LINES = 66673 # wc -l
NUM_PROCESS = 3

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
    

def data_generator(filename, queue, start_position, end_position, gel: GenereteErrorLine, tokenizer):
    counter = 0 
    with open(filename, 'r') as f:
        while counter != start_position:
            f.readline()
            counter += 1

        while counter != end_position:
            line = f.readline()
            
            error_line = gel(line)
            
            tokenized = tokenizer(error_line, text_target=line, truncation=True, return_tensors="tf")
            
            # input_ids = tf.reshape(tokenized['input_ids'][0], (-1))
            # attention_mask = tf.reshape(tokenized['attention_mask'][0], (-1))
            # labels = tf.reshape(tokenized['labels'][0], (-1))

            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]
            labels = tokenized['labels'][0]

            queue.put((input_ids, attention_mask, labels))
            
            counter += 1

            if not line: # EOF
                f.seek(0) 
                counter = 0

def run_processes(queue: Queue, num_parallel: int, filename: str, file_size: int, gel: GenereteErrorLine, tokenizer):
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
        process = Process(target=data_generator, args=(filename, queue, positions[i], positions[i+1], gel, tokenizer,))
        process.start()
        processes.append(process)

    return processes


def read_preproccess_data(queue: Queue, files: List[str], file_sizes: List[int], num_parallel: int, gel: GenereteErrorLine, tokenizer):
    index = 0
    while True:
        file = files[index]
        file_size = file_sizes[index]
        
        processes = run_processes(queue, num_parallel, file, file_size, gel, tokenizer)
        for process in processes:
            process.join()

        index += 1
        if index == len(files):
            index = 0


tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

lang = "cs"
token_file = "/home/petr/Plocha/DP/czech-gec/code/data/vocabluraries/vocabulary_cs.tsv"
tokens = introduce_errors.get_token_vocabulary(token_file)
characters = introduce_errors.get_char_vocabulary(lang)
aspell_speller = aspell.Speller('lang', lang)
token_err_distribution = [0.7, 0.1, 0.1, 0.1, 0]
char_err_distribution = [0.25, 0.25, 0.25, 0.25, 0]
token_err_prob = 0.15
char_err_prob = 0.02


random.seed(42)
queue = Queue(2 * NUM_PROCESS)
gel = GenereteErrorLine(tokens, characters, aspell_speller, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob)

process = Process(target=read_preproccess_data, args=(queue, [PATH], [NUM_LINES], NUM_PROCESS, gel, tokenizer,))
process.start()

dataset = tf.data.Dataset.from_generator(
    lambda: iter(queue.get, None),
    output_types=(tf.int32, tf.int32, tf.int32),
    output_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))

dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=4))
# dataset = dataset.ragged_batch(4)
dataset = dataset.prefetch(2) # Number of batches to prefetch


for d in dataset:
    print(d)

import string
import aspell
import errant
import random
import argparse
import numpy as np

# from edit import Edit
from .edit import Edit
from typing import List
from itertools import compress
from abc import ABC, abstractmethod
from errant.annotator import Annotator

czech_diacritics_tuples = [('a', 'á'), ('c', 'č'), ('d', 'ď'), ('e', 'é', 'ě'), ('i', 'í'), ('n', 'ň'), ('o', 'ó'), ('r', 'ř'), ('s', 'š'),
                           ('t', 'ť'), ('u', 'ů', 'ú'), ('y', 'ý'), ('z', 'ž')]
czech_diacritizables_chars = [char for sublist in czech_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                  czech_diacritics_tuples for char in
                                                                                                  sublist]


class Error(ABC):
    def __init__(self, target_prob: float) -> None:
        self.target_prob = target_prob
        self.num_errors = 0
        self.num_possible_edits = 0

    @abstractmethod
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        pass


class ErrorMeMne(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == "mně":
                c_toks = annotator.parse("mě")
                edit = Edit(token, c_toks, [i, i+1, i, i+1], type="MeMne")
                edits.append(edit)
            if token.text == "mě":
                c_toks = annotator.parse("mně")
                edit = Edit(token, c_toks, [i, i+1, i, i+1], type="MeMne")
                edits.append(edit)
        return edits


class ErrorReplace(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.isalpha():
                proposals = aspell_speller.suggest(token.text)[:10]
                if len(proposals) > 0:
                    new_token_text = np.random.choice(proposals)
                    c_toks = annotator.parse(new_token_text)
                    edit = Edit(token, c_toks, [i, i+1, i, i+1], type="Replace")
                    edits.append(edit)
        return edits


class ErrorInsert(Error):
    def __init__(self, target_prob: float, word_vocabulary) -> None:
        super().__init__(target_prob)
        self.word_vocabulary = word_vocabulary

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            new_token_text = np.random.choice(self.word_vocabulary)
            c_toks = annotator.parse(new_token_text)
            edit = Edit(token, c_toks, [i, i, i, i+1], type="Insert")
            edits.append(edit)
        return edits


class ErrorDelete(Error):
    def __init__(self, target_prob: float) -> None:
        super().__init__(target_prob)
        self.allowed_source_delete_tokens = [',', '.', '!', '?']

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.isalpha() and token.text not in self.allowed_source_delete_tokens:
                c_toks = annotator.parse("")
                edit = Edit(token, c_toks, [i, i+1, i, i], type="Remove")
                edits.append(edit)
        return edits


class ErrorRecase(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.islower():
                new_token_text = token.text[0].upper() + token.text[1:]
            else:
                num_recase = min(len(token.text), max(1, int(np.round(np.random.normal(0.3, 0.4) * len(token.text)))))
                char_ids_to_recase = np.random.choice(len(token.text), num_recase, replace=False)
                new_token_text = ''
                for char_i, char in enumerate(token.text):
                    if char_i in char_ids_to_recase:
                        if char.isupper():
                            new_token_text += char.lower()
                        else:
                            new_token_text += char.upper()
                    else:
                        new_token_text += char
            c_toks = annotator.parse(new_token_text)
            edit = Edit(token, c_toks, [i, i+1, i, i+1], type="Recase")
            edits.append(edit)
        return edits


class ErrorSwap(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            previous_token = parsed_sentence[0]
            for i, token in enumerate(parsed_sentence[1:]):
                i = i + 1
                c_toks = annotator.parse(token.text + " " + previous_token.text)
                edit = Edit(token, c_toks, [i-1, i+1, i-1, i+1], type="Swap")
                edits.append(edit)
                previous_token = token
        return edits


# MAIN:
class ErrorGenerator:
    def __init__(self, lang: str, char_level_params, word_vocabulary) -> None:
        self.replace_prob = char_level_params[0]
        self.insert_prob = char_level_params[1]
        self.delete_prob = char_level_params[2]
        self.swap_prob = char_level_params[3]
        self.change_diacritics_prob = char_level_params[4]
        self.err_prob = char_level_params[5]
        self.std_dev = char_level_params[6]
        self.char_vocabulary = char_level_params[7]

        self.annotator = None

        self.total_tokens = 0
        self.error_instances = [
            ErrorMeMne(
                0.0000),
            ErrorReplace(
                0.1050),
            ErrorInsert(
                0.0150, word_vocabulary),
            ErrorDelete(
                0.0075),
            ErrorRecase(
                0.0150),
            ErrorSwap(
                0.0075)
        ]

    # def get_edits(self, parsed_sentence) -> List[Edit]:
    #     self.total_tokens += len(parsed_sentence)
    #     all_edits = []
    #     for error_instance in self.error_instances:
    #         edits = error_instance(parsed_sentence, self.annotator)
    #         ## rejection sampling
    #         selected_edits = []
    #         for edit in edits:
    #             gen_prob = error_instance.num_possible_edits / self.total_tokens if self.total_tokens > 0 else 0.5
    #             acceptance_prob = error_instance.target_prob / (gen_prob + 1e-10)
    #             if np.random.uniform(0, 1) < acceptance_prob:
    #                 selected_edits.append(edit)
    #                 error_instance.num_errors += 1
    #         error_instance.num_possible_edits += len(edits)
    #         ##
    #         all_edits = all_edits + selected_edits
    #     # TODO: Do not accept all edits.
    #     return all_edits

    def _init_annotator(self, lang: str = 'cs'):
        if self.annotator is None:
            self.annotator = errant.load(lang)

    def get_edits(self, parsed_sentence, annotator: Annotator, aspell_speller) -> List[Edit]:
        self.total_tokens += len(parsed_sentence)
        edits_errors = []
        for error_instance in self.error_instances:
            edits = error_instance(parsed_sentence, annotator, aspell_speller)
            edits_errors = edits_errors + [(edit, error_instance) for edit in edits]
        
        if len(edits_errors) == 0:
            return []

        # Overlaping:
        random.shuffle(edits_errors)
        mask = self.get_remove_mask(list(zip(*edits_errors))[0])
        edits_errors = list(compress(edits_errors, mask))
        
        ## Rejection Sampling:
        selected_edits = []
        for edit, error_instance in edits_errors:
            gen_prob = error_instance.num_possible_edits / self.total_tokens if self.total_tokens > 0 else 0.5
            acceptance_prob = error_instance.target_prob / (gen_prob + 1e-10)
            if np.random.uniform(0, 1) < acceptance_prob:
                selected_edits.append(edit)
                error_instance.num_errors += 1
        error_instance.num_possible_edits += len(edits_errors)
        ##

        # Sorting:
        sorted_edits = self.sort_edits_reverse(selected_edits)
        return sorted_edits
    
    def sort_edits_reverse(self, edits: List[Edit]) -> List[Edit]:
        minus_start_indices = [(-1) * edit.o_end for edit in edits]
        sorted_edits = np.array(edits)
        sorted_edits = sorted_edits[np.argsort(minus_start_indices)]

        minus_start_indices = [(-1) * edit.o_start for edit in sorted_edits]
        sorted_edits = np.array(sorted_edits)
        sorted_edits = sorted_edits[np.argsort(minus_start_indices)]

        return sorted_edits.tolist()


    def get_remove_mask(self, edits: List[Edit]) -> List[bool]:
        ranges = [(edit.o_start, edit.o_end) for edit in edits]
        removed = [not any([self.is_overlap(current_range, r) if j < i else False for j, r in enumerate(ranges)]) for i, current_range in enumerate(ranges)]
        # filtered_edits = list(compress(edits, removed))
        return removed

    def is_overlap(self, range_1: tuple, range_2: tuple) -> bool:
        start_1 = range_1[0]
        end_1 = range_1[1]
        start_2 = range_2[0]
        end_2 = range_2[1]

        if start_1 <= start_2:
            if end_1 > start_2:
                return True
        else:
            if end_2 > start_1:
                return True
        return False
    
    def get_m2_edits_text(self, sentence: str, annotator: Annotator, aspell_speller) -> List[str]:
        parsed_sentence = annotator.parse(sentence)
        edits = self.get_edits(parsed_sentence, annotator, aspell_speller)
        m2_edits = [edit.to_m2() for edit in edits]
        return m2_edits
    
    def introduce_char_level_errors_on_sentence(self, sentence):
        replace_prob = self.replace_prob
        insert_prob = self.insert_prob
        delete_prob = self.delete_prob
        swap_prob = self.swap_prob
        change_diacritics_prob = self.change_diacritics_prob
        err_prob = self.err_prob
        std_dev = self.std_dev
        char_vocabulary = self.char_vocabulary

        sentence = list(sentence)
        num_errors = int(np.round(np.random.normal(err_prob, std_dev) * len(sentence)))
        num_errors = min(max(0, num_errors), len(sentence))  # num_errors \in [0; len(sentence)]
        if num_errors == 0:
            return ''.join(sentence)
        char_ids_to_modify = np.random.choice(len(sentence), num_errors, replace=False)
        new_sentence = ''
        for char_id in range(len(sentence)):
            if char_id not in char_ids_to_modify:
                new_sentence += sentence[char_id]
                continue
            operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'change_diacritics'], 1,
                                         p=[replace_prob, insert_prob, delete_prob, swap_prob, change_diacritics_prob])
            current_char = sentence[char_id]
            new_char = ''
            if operation == 'replace':
                if current_char.isalpha():
                    new_char = np.random.choice(char_vocabulary)
                else:
                    new_char = current_char
            elif operation == 'insert':
                new_char = current_char + np.random.choice(char_vocabulary)
            elif operation == 'delete':
                if current_char.isalpha():
                    new_char = ''
                else:
                    new_char = current_char
            elif operation == 'swap':
                if char_id == len(sentence) - 1:
                    continue
                new_char = sentence[char_id + 1]
                sentence[char_id + 1] = sentence[char_id]
            elif operation == 'change_diacritics':
                if current_char in czech_diacritizables_chars:
                    is_lower = current_char.islower()
                    current_char = current_char.lower()
                    char_diacr_group = [group for group in czech_diacritics_tuples if current_char in group][0]
                    new_char = np.random.choice(char_diacr_group)
                    if not is_lower:
                        new_char = new_char.upper()
            new_sentence += new_char
        return new_sentence
    
    def create_error_sentence(self, sentence: str, aspell_speller, use_char_level: bool = False) -> List[str]:
        # annotator = errant.load('cs')
        try:
            print("BEFORE")
            parsed_sentence = self.annotator.parse(sentence)
            print("AFTER")
        except Exception as e:
            print(e)
        edits = self.get_edits(parsed_sentence, self.annotator, aspell_speller)
        # TODO: sort sem (aby m3 format byl spravne)
        for edit in edits:
            start, end = edit.o_start, edit.o_end
            cor_toks_str = " ".join([tok.text for tok in edit.c_toks])
            # TODO: Do it better
            sentence = parsed_sentence[:start].text + " " + cor_toks_str + " " + parsed_sentence[end:].text
            parsed_sentence = self.annotator.parse(sentence)
        sentence = parsed_sentence.text
        
        if use_char_level:
            sentence = self.introduce_char_level_errors_on_sentence(sentence)

        return sentence


def get_token_vocabulary(tsv_token_file):
    tokens = []
    with open(tsv_token_file) as reader:
        for line in reader:
            line = line.strip('\n')
            token, freq = line.split('\t')
            if token.isalpha():
                tokens.append(token)
    return tokens

def get_char_vocabulary(lang):
    if lang == 'cs':
        czech_chars_with_diacritics = 'áčďěéíňóšřťůúýž'
        czech_chars_with_diacritics_upper = czech_chars_with_diacritics.upper()
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + czech_chars_with_diacritics + czech_chars_with_diacritics_upper
        return list(allowed_chars)


def main(args):
    characters = get_char_vocabulary(args.lang)
    char_level_params = (
        0.2, 0.2, 0.2, 0.2, 0.2, 0.02, 0.01, characters
    )
    word_vocabulary = get_token_vocabulary("../../data/vocabluraries/vocabulary_cs.tsv")
    aspell_speller = aspell.Speller('lang', args.lang)
    annotator = errant.load(args.lang)
    error_generator = ErrorGenerator(args.lang, char_level_params, word_vocabulary)
    input_path = args.input
    output_path = args.output
    with open(input_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()

            # m2_lines = error_generator.get_m2_edits_text(line, annotator, aspell_speller)
            # with open(output_path, "a+") as output_file:
            #     output_file.write("S " + line + "\n")
            #     for m2_line in m2_lines:
            #         output_file.write(m2_line + "\n")
            #     output_file.write("\n")

            error_line = error_generator.create_error_sentence(line, annotator, aspell_speller, True)
            with open(output_path, "a+") as output_file:
                output_file.write(error_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create m2 file with errors.")
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default="output.m2")
    parser.add_argument('-l', '--lang', type=str)

    args = parser.parse_args()
    main(args)

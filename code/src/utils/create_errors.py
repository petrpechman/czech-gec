import errant
import argparse
import numpy as np

from edit import Edit
from typing import List
from abc import ABC, abstractmethod
from errant.annotator import Annotator

class Error(ABC):
    def __init__(self, target_prob: float) -> None:
        self.target_prob = target_prob
        self.num_errors = 0
        self.num_possible_edits = 0

    @abstractmethod
    def __call__(self, parsed_sentence, annotator: Annotator) -> List[Edit]:
        pass


class ErrorMeMne(Error):
    def __call__(self, parsed_sentence, annotator: Annotator) -> List[Edit]:
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


class ErrorGenerator:
    def __init__(self, lang: str) -> None:
        self.annotator = errant.load(args.lang)
        self.total_tokens = 0
        self.error_instances = [
            ErrorMeMne(0.125)
        ]

    def get_edits(self, parsed_sentence) -> List[Edit]:
        self.total_tokens += len(parsed_sentence)
        all_edits = []
        for error_instance in self.error_instances:
            edits = error_instance(parsed_sentence, self.annotator)
            ## rejection sampling
            selected_edits = []
            for edit in edits:
                gen_prob = error_instance.num_possible_edits / self.total_tokens if self.total_tokens > 0 else 0.5
                acceptance_prob = error_instance.target_prob / (gen_prob + 1e-10)
                if np.random.uniform(0, 1) < acceptance_prob:
                    selected_edits.append(edit)
                    error_instance.num_errors += 1
            error_instance.num_possible_edits += len(edits)
            ##
            all_edits = all_edits + selected_edits
        # TODO: Do not accept all edits.
        return all_edits
    
    def get_m2_edits_text(self, sentence: str) -> List[str]:
        parsed_sentence = self.annotator.parse(sentence)
        edits = self.get_edits(parsed_sentence)
        m2_edits = [edit.to_m2() for edit in edits]
        return m2_edits
    
    def create_error_sentence(self, sentence: str) -> List[str]:
        parsed_sentence = self.annotator.parse(sentence)
        edits = self.get_edits(parsed_sentence)
        for edit in edits:
            start, end = edit.o_start, edit.o_end
            cor_toks_str = " ".join([tok.text for tok in edit.c_toks])
            # TODO: Do it better
            sentence = parsed_sentence[:start].text + " " + cor_toks_str + " " + parsed_sentence[end:].text
            parsed_sentence = self.annotator.parse(sentence)
        return parsed_sentence.text


def main(args):
    error_generator = ErrorGenerator(args.lang) 
    input_path = args.input
    output_path = args.output
    with open(input_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            m2_lines = error_generator.get_m2_edits_text(line)
            with open(output_path, "a+") as output_file:
                output_file.write("S " + line + "\n")
                for m2_line in m2_lines:
                    output_file.write(m2_line + "\n")
                output_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create m2 file with errors.")
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default="output.m2")
    parser.add_argument('-l', '--lang', type=str)

    args = parser.parse_args()
    main(args)

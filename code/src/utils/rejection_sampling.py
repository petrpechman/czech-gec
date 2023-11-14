from typing import List
import numpy as np


def get_edits(sentence) -> List:
    if np.random.uniform() < 0.5:
        if np.random.uniform() < 0.1:
            return ["edit", "edit"]
        else:
            return ["edit"]
    else:
        return []
    
def get_tokens(sentence: str) -> List:
    return sentence.split(' ')

gen_prob = 0.5
tar_prob = 0.014

total_tokens = 0
num_errors = 0


sentences = ["Moje krasne super veta"] * 100000


for i, sentence in enumerate(sentences):
    edits = get_edits(sentence)
    tokens = get_tokens(sentence)

    for edit in edits:
        if tar_prob > gen_prob:
            num_errors += 1
    
    total_tokens += len(tokens)

    gen_prob = num_errors / total_tokens
        
print(gen_prob)

print("TOT", total_tokens)
print("ERR", num_errors)
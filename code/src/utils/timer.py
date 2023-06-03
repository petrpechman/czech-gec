from introduce_errors import introduce_token_level_errors_on_sentence, introduce_token_level_errors_on_sentence_timer, get_char_vocabulary, get_token_vocabulary, introduce_char_level_errors_on_sentence
import aspell
import numpy as np
from timeit import timeit

token_file = "../../data/vocabluraries/vocabulary_cs.tsv"
tokens = get_token_vocabulary(token_file)
characters = get_char_vocabulary("cs")
token_err_distribution = [0.7, 0.1, 0.1, 0.1, 0]
char_err_distribution = [0.25, 0.25, 0.25, 0.25, 0]
token_err_prob = 0.15
token_std_dev = 0.2
char_err_prob = 0.02 
char_std_dev=0.01   
aspell_speller = aspell.Speller('lang', "cs")

token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = token_err_distribution
char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob = char_err_distribution

seed = 1412

np.random.seed(seed)
times = [0, 0, 0, 0, 0, 0, 0]
for i in range(1000):
    result, r_times = introduce_token_level_errors_on_sentence_timer(
                                                        "Moje krásná věta .".split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, token_err_prob, token_std_dev,
                                                        tokens, aspell_speller)

    for j in range(len(times)):
        times[j] += r_times[j]

print(times)

np.random.seed(seed)
result1 = introduce_token_level_errors_on_sentence_timer("Moje krásná věta .".split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, token_err_prob, token_std_dev,
                                                        tokens, aspell_speller)
time1 = timeit(lambda: introduce_token_level_errors_on_sentence_timer("Moje krásná věta .".split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, token_err_prob, token_std_dev,
                                                        tokens, aspell_speller), number=1000)


np.random.seed(seed)
time2 = timeit(lambda: introduce_char_level_errors_on_sentence("Moje krásná věta .", char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob,
                                                       change_diacritics_prob, char_err_prob, char_std_dev,
                                                       characters), number=1000)                                                    
np.random.seed(seed)
result3 = introduce_token_level_errors_on_sentence("Moje krásná věta .".split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, token_err_prob, token_std_dev,
                                                        tokens, aspell_speller)
time3 = timeit(lambda: introduce_token_level_errors_on_sentence("Moje krásná věta .".split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, token_err_prob, token_std_dev,
                                                        tokens, aspell_speller), number=1000)
print(result1)
print(result3)


print(time1)
print(time2)
print(time3)
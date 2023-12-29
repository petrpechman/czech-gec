import numpy as np

# from edit import Edit
from .edit import Edit
from typing import List
from abc import ABC, abstractmethod
from errant.annotator import Annotator

class Error(ABC):
    def __init__(self, target_prob: float, absolute_prob: float = 0.0, use_absolute_prob: bool = False) -> None:
        self.target_prob = target_prob
        self.absolute_prob = absolute_prob
        self.use_absolute_prob = use_absolute_prob
        self.num_errors = 0
        self.num_possible_edits = 0

    @abstractmethod
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        pass

    def _capitalize_words(self, variants: List) -> List:
        capital_variants = []
        for word, possible_errors, use_capital in variants:
            if use_capital:
                capital_possible_errors = []
                for possible_error in possible_errors:
                    possible_error.capitalize()
                capital_variants.append((word.capitalize(), capital_possible_errors))
        return capital_variants

    def _general(self, variants: List, parsed_sentence, annotator: Annotator, type: str) -> List[Edit]:
        first_word_variants = self._capitalize_words(variants)
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i == 0:
                for (right, possible_errors) in first_word_variants:
                    if token.text == right:
                        o_toks = annotator.parse(token.text)
                        for possible_error in possible_errors:
                            c_toks = annotator.parse(possible_error)
                            edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type=type)
                            edits.append(edit)
            for (right, possible_errors, _) in variants:
                if token.text == right:
                    o_toks = annotator.parse(token.text)
                    for possible_error in possible_errors:
                        c_toks = annotator.parse(possible_error)
                        edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type=type)
                        edits.append(edit)
        return edits


class ErrorMeMne(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == "mně":
                o_toks = annotator.parse("mně")
                c_toks = annotator.parse("mě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                edits.append(edit)
            elif token.text == "mě":
                o_toks = annotator.parse("mě")
                c_toks = annotator.parse("mně")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                edits.append(edit)
            if i == 0:
                if token.text == "Mně":
                    o_toks = annotator.parse("Mně")
                    c_toks = annotator.parse("Mě")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                    edits.append(edit)
                elif token.text == "Mě":
                    o_toks = annotator.parse("Mě")
                    c_toks = annotator.parse("Mně")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                    edits.append(edit)
        return edits
    
class ErrorMeMneSuffix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and token.text.endswith("mně"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-3] + "mě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneSuffix")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.endswith("mě"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-2] + "mně")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneSuffix")
                edits.append(edit)
        return edits
    
class ErrorMeMneIn(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and 'mně' in token.text and not token.text.endswith("mně"):
                index = token.text.find('mně')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "mě" + token.text[index+3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 2 and 'mě' in token.text and not token.text.endswith("mě"):
                index = token.text.find('mě')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "mně" + token.text[index+2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.startswith('Mě'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Mně" + token.text[2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 3 and token.text.startswith('Mně'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Mě" + token.text[3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
        return edits
    
class ErrorDNT(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        specific_chars = ['d', 't', 'n']
        edits = []
        for i, token in enumerate(parsed_sentence):
            for specific_char in specific_chars:
                if specific_char + 'i' in token.text:
                    index = token.text.find(specific_char + 'i')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'y' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                elif specific_char + 'í' in token.text:
                    index = token.text.find(specific_char + 'í')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'ý' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                if specific_char + 'y' in token.text:
                    index = token.text.find(specific_char + 'y')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'i' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                elif specific_char + 'ý' in token.text:
                    index = token.text.find(specific_char + 'ý')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'í' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
        return edits
    
class ErrorEnumeratedWord(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        specific_chars = ['b', 'f', 'l', 'm', 'p', 's', 'v', 'z']
        edits = []
        for i, token in enumerate(parsed_sentence):
            for specific_char in specific_chars:
                if specific_char + 'i' in token.text:
                    index = token.text.find(specific_char + 'i')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'y' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                elif specific_char + 'í' in token.text:
                    index = token.text.find(specific_char + 'í')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'ý' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                if specific_char + 'y' in token.text:
                    index = token.text.find(specific_char + 'y')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'i' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                elif specific_char + 'ý' in token.text:
                    index = token.text.find(specific_char + 'ý')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'í' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
        return edits

class ErrorSuffixIY(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.endswith("y"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "i")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("ý"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "í")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("i") and len(token.text) > 1:
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "y")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("í"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "ý")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
        return edits
    
class ErrorUU(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if 'ů' in token.text:
                index = token.text.find('ů')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "ú" + token.text[index+1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
            elif 'ú' in token.text:
                index = token.text.find('ú')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "ů" + token.text[index+1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
            elif token.text.startswith('Ú'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Ů" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
        return edits

class ErrorCondional(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.variants = [
            ("bychom", ["bysme", "by jsme"], True),
            ("byste", ["by ste", "by jste", "by jsi"], True),
            ("bych", ["by jsem", "bysem"], True),
            ("bys", ["by jsi", "by si"], True),
            ("abybychom", ["abybysme", "aby jsme"], True),
            ("abybyste", ["aby ste", "aby jste", "aby jsi"], True),
            ("abybych", ["aby jsem", "abysem"], True),
            ("abybys", ["aby jsi", "aby si"], True),
            ("kdybychom", ["kdybysme", "kdyby jsme"], True),
            ("kdybyste", ["kdyby ste", "kdyby jste", "kdyby jsi"], True),
            ("kdybych", ["kdyby jsem", "kdybysem"], True),
            ("kdybys", ["kdyby jsi", "kdyby si"], True),
        ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "Conditional")
    
class ErrorSpecificWords(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.variants = [
            ("viz", ["viz."], False),
            ("výjimka", ["vyjímka"], True),
            ("seshora", ["zeshora", "zezhora"], True),
        ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "SpecificWords")
    
class ErrorSZPrefix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 1 and token.text.startswith("s") and token.text != "se":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("z" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("z") and token.text != "ze":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("s" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("S") and token.text != "Se":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Z" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("Z") and token.text != "Ze":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("S" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
        return edits

class ErrorNumerals(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.variants = [
            ("oběma", ["oběmi, oboumi", "obouma"], True),
            ("dvěma", ["dvěmi", "dvouma"], True),
            ("třemi", ["třema"], True),
            ("čtyřmi", ["čtyřma"], True),
        ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "Numerals")
    
class ErrorMyMi(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == "mi":
                o_toks = annotator.parse("mi")
                c_toks = annotator.parse("my")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                edits.append(edit)
            elif token.text == "my":
                o_toks = annotator.parse("my")
                c_toks = annotator.parse("mi")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                edits.append(edit)
            if i == 0:
                if token.text == "Mi":
                    o_toks = annotator.parse("Mi")
                    c_toks = annotator.parse("My")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                    edits.append(edit)
                elif token.text == "My":
                    o_toks = annotator.parse("My")
                    c_toks = annotator.parse("Mi")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                    edits.append(edit)
        return edits
    
class ErrorBeBjeSuffix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and token.text.endswith("bje"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-3] + "bě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeSuffix")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.endswith("bě"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-2] + "bje")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeSuffix")
                edits.append(edit)
        return edits
    
class ErrorBeBjeIn(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and 'bje' in token.text and not token.text.endswith("bje"):
                index = token.text.find('bje')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "bě" + token.text[index+3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 2 and 'bě' in token.text and not token.text.endswith("bě"):
                index = token.text.find('bě')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "bje" + token.text[index+2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.startswith('Bě'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Bje" + token.text[2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 3 and token.text.startswith('Bje'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Bě" + token.text[3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
        return edits
    
class ErrorSebou(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        prev_token = None
        for i, token in enumerate(parsed_sentence):
            if 'sebou' == token.text:
                if prev_token and prev_token == "s":
                    o_toks = annotator.parse("s sebou")
                    c_toks = annotator.parse("sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
                else:
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("s sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
            if i == 0:
                if 'Sebou' == token.text:
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("S sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
            if i == 1:
                if prev_token == 'S' and token.text == 'sebou':
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("Sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
        return edits

class ErrorSentenceFirstUpper(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            token = parsed_sentence[0]
            if token.text.istitle():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.lower())
                edit = Edit(o_toks, c_toks, [0, len(o_toks), 0, len(c_toks)], type="SentenceFirstUpper")
                edits.append(edit)
        return edits

class ErrorSentenceFirstLower(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            token = parsed_sentence[0]
            if token.text.islower():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.title())
                edit = Edit(o_toks, c_toks, [0, len(o_toks), 0, len(c_toks)], type="SentenceFirstLower")
                edits.append(edit)
        return edits
    
class ErrorTitleToLower(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i > 0 and token.text.istitle():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.lower())
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="TitleToLower")
                edits.append(edit)
        return edits
    
class ErrorLowerToTitle(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i > 0 and token.text.islower():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.title())
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="LowerToTitle")
                edits.append(edit)
        return edits

class ErrorPrepositionSZ(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.variants = [
            ("s", ["se"], True),
            ("z", ["ze"], True),
            ("se", ["s"], True),
            ("ze", ["z"], True),
        ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "PrepositionSZ")

ERRORS = {
    "MeMne": ErrorMeMne,
    "MeMneSuffix": ErrorMeMneSuffix,
    "MeMneIn": ErrorMeMneIn,
    "DTN": ErrorDNT,
    "SuffixIY": ErrorSuffixIY,
    "UU": ErrorUU,
    "Conditional": ErrorCondional,
    "SpecificWords": ErrorSpecificWords,
    "SZPrefix": ErrorSZPrefix,
    "Numerals": ErrorNumerals,
    "MyMi": ErrorMyMi,
    "BeBjeSuffix": ErrorBeBjeSuffix,
    "BeBjeIn": ErrorBeBjeIn,
    "Sebou": ErrorSebou,
    "SentenceFirstUpper": ErrorSentenceFirstUpper,
    "SentenceFirstLower": ErrorSentenceFirstLower,
    "TitleToLower": ErrorTitleToLower,
    "LowerToTitle": ErrorLowerToTitle,
    "PrepositionSZ": ErrorPrepositionSZ,
}

### NO USED:

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

###
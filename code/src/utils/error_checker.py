import errant
from errant.edit import Edit
import csv

class ErrorChecker:
    def __init__(self, filename: str = "typical_errors.tsv", log_every_nth: int = 10):
        self.filename = filename
        self.data = dict()
        self.data['total'] = 0
        self.annotator = errant.load('cs')
        self.counter = 0
        self.log_every_nth = log_every_nth


    def __call__(self, original_sentence: str, correct_sentence: str) -> None:
        edits = self.get_edits(original_sentence, correct_sentence)
        for edit in edits:
            self.data['total'] += 1
            self._check_error_mne(edit)
        
        self.counter += 1
        if self.counter == self.log_every_nth:
            self.save_into_file()
            self.counter = 0


    def get_edits(self, orig: str, cor: str) -> list:
        orig = orig.strip()
        cor = cor.strip()
        orig = self.annotator.parse(orig, False)
        if orig.text.strip() == cor:
            return []
        else:
            cor = self.annotator.parse(cor, False)
            edits = self.annotator.annotate(orig, cor, False, "rules")
            return edits
        
    def save_into_file(self):
        with open(self.filename, 'w') as csv_file:  
            writer = csv.writer(csv_file, delimiter='\t')
            for key, value in self.data.items():
                writer.writerow([key, value])


    def _add_error_type(self, error_type: str):
        if error_type in self.data: 
            self.data[error_type] += 1
        else:
            self.data[error_type] = 1


    def _check_error_mne(self, edit: Edit): 
        if edit.o_str == "mě" and \
           edit.c_str == "mně" and \
           edit.c_start + 1 == edit.c_end and \
           edit.o_start + 1 == edit.o_end:
            self._add_error_type('mne')

origins = [
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
    "Přijď ke mě",
]

corrects = [
    "Přijď ke mně",
    "Přijď ke mně",
    "Přijď ke mně",
    "Přijď ke mně",
    "Přijď ke mě taky",
    "Přijď ke mě taky",
    "Přijď ke mě taky",
    "Přijď ke mě taky",
]

ec = ErrorChecker('./file.tsv')
for o, c in zip(origins, corrects):
    ec(o, c)
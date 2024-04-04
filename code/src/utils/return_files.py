import os

current_dir = os.getcwd()
filenames = os.listdir(current_dir)
filenames = [filename for filename in filenames if not filename.startswith('ckpt-')]
filenames = [filename for filename in filenames if filename.split('-')[-1].isdigit()]

for filename in filenames:
    number = filename.split('-')[-1]
    last_index = filename.rfind('-')
    new_name = number + '-' + filename[:last_index]
    print(new_name)
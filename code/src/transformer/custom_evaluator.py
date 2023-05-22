import os
import time
from transformers import TFAutoModelForSeq2SeqLM, AutoConfig
# import tensorflow as tf

checkpoint_dir = "./tmp/checkpoint/"
MODEL = "../utils/transformer/model/"
FROM_CONFIG = False

# Initialize a set to store the paths of already processed checkpoint files
processed_files = set()

if FROM_CONFIG:
    config = AutoConfig.from_pretrained(MODEL)
    model = TFAutoModelForSeq2SeqLM.from_config(config)
else:
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL)

if loss:
    model.compile(optimizer=optimizer, loss=loss, metrics=[M2Scorer(dev_gold_edits, None)])
else:
    model.compile(optimizer=optimizer, metrics=[M2Scorer(dev_gold_edits, None)])

model.load_weights(filepath=checkpoint_dir)

model.generate(["Neco tu je napsane..."])

# while True:
#     file_list = os.listdir(checkpoint_dir)
#     print("processing checkpoint")
#     for file in file_list:
#         if file.startswith("ckpt-"):
#             file_path = os.path.join(checkpoint_dir, file)
#             if file_path not in processed_files: # must be folder (/ at the end)
#                 # model.load_weights(checkpoint_dir)
#                 print(file_path)
#                 processed_files.add(file_path)
    
#     # Wait for a specific interval before checking for new files again
#     time.sleep(60)  # Wait for 60 seconds before checking again

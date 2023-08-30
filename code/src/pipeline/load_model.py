import argparse
import sys
sys.path.append('..')

import os
import json
import tensorflow as tf

from transformers import AutoConfig
from transformers import TFAutoModelForSeq2SeqLM

BACKUP_DIR = "../bart/tmp/backup/"
MODEL = "../../models/transformer/"

config = AutoConfig.from_pretrained(MODEL)
model = TFAutoModelForSeq2SeqLM.from_config(config)


checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(BACKUP_DIR))

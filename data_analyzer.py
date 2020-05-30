import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np
import math
import sys
import collections

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from tensorflow.data.experimental import AUTOTUNE

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', action='store',
                      default=64, dest='batch_size',
                      help='Training batch size', type=int)
parser.add_argument('--dataset_path', action='store',
                      default='~/tensorflow_datasets/', dest='dataset_path',
                      help='Path to store input data.')
                      
args = parser.parse_args()

main_ds, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=args.dataset_path)
val_ds, test_ds = tfds.load('oxford_iiit_pet:3.*.*', split=['test[:50%]', 'test[-50%:]'], data_dir=args.dataset_path)
print(info)

PIXEL_DATA_TYPE = tf.float32
def normalize(input_image, input_mask, num_classes=3):
  input_image = tf.cast(input_image, PIXEL_DATA_TYPE) / 255.0
  # Onehot encode the class masks, i.e. turn 0 into [1,0,0] and 1 into [0,1,0] and 2 into [0,0,1]
  input_mask = tf.squeeze(tf.one_hot(indices=tf.cast(input_mask, dtype=tf.uint8), depth=num_classes), axis=-2)
  return input_image, input_mask
  
@tf.function
def load_image_train(datapoint):
  print('Loading training images')

  input_image, input_mask = normalize(datapoint['image'], datapoint['segmentation_mask'])

  return input_image, input_mask

TRAIN_LENGTH = info.splits['train'].num_examples
BUFFER_SIZE = args.batch_size * 4
STEPS_PER_EPOCH = TRAIN_LENGTH // args.batch_size

###############################################################################
# Count stuff
###############################################################################
max_pics = 50
num_pics = 0
counter = collections.Counter()
for item in main_ds['train']:
  for mask_pixel_type in K.flatten(item['segmentation_mask']):
    counter[K.get_value(mask_pixel_type)] += 1
  num_pics += 1
  print(num_pics)
  if num_pics >= max_pics:
    break

print('Pixel frequency counts:')
print(counter)
  

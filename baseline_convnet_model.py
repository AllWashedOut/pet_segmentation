import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np
import math
import sys

import keras
from keras.models import load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, BatchNormalization
from keras import backend as K
from tensorflow.data.experimental import AUTOTUNE

from clr_callback import CyclicLR

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', action='store',
                      default=64, dest='batch_size',
                      help='Training batch size', type=int)
parser.add_argument('--patience', action='store',
                      default=20, dest='patience',
                      help='Training patience', type=int)
parser.add_argument('--model_path', action='store',
                      default='segment.h5', dest='model_path',
                      help='Output path for the saved model')
parser.add_argument('--dataset_path', action='store',
                      default='~/tensorflow_datasets/', dest='dataset_path',
                      help='Path to store input data.')
parser.add_argument('--max_epochs', action='store',
                      default=75, dest='max_epochs',
                      help='Training max_epochs', type=int)
parser.add_argument('--image_size', action='store',
                      default=128, dest='image_size',
                      help='Square resize all input images to this many pixels per dimension', type=int)
parser.add_argument('--continue_existing', action='store_true',
                      default=False, dest='continue_existing',
                      help='Load an existing model and continue training')
parser.add_argument('--cyclical_learning_rate', action='store_true',
                      default=False, dest='cyclical_learning_rate',
                      help='Vary the learning rate in a cycle')
                      
args = parser.parse_args()

###############################################################################
# Loss
###############################################################################

CCE = keras.losses.CategoricalCrossentropy(from_logits=True)

###############################################################################
# Data augmentation
# Vary pixel color/luminance and take random crops.
# Some care is taken to do this at every iteration, rather than only once per
# test run.
###############################################################################
def colorAugmentations(img, mask):
  # None
  return img, mask

def geometryAugmentations(img, mask):
  # Augmentations that alter geometry. That means that they
  # should be applied to image AND mask.
  # Some random operations are done on a concatenated image (img & mask)
  # so that the effect is equal on both.
  combined = tf.concat([img, mask], axis=2)
  combined = tf.image.random_flip_left_right(combined)
  
  img = combined[:, :, :3]
  mask = combined[:, :, 3:]
  
  img = tf.image.resize(img, (args.image_size, args.image_size))
  # Masks should not be interpolated, that would give non-integer values.
  mask = tf.image.resize(mask, (args.image_size, args.image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  return img, mask

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
  
  #input_image = preprocess_input(input_image)

  return input_image, input_mask
  
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (args.image_size, args.image_size))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (args.image_size, args.image_size))

  input_image, input_mask = normalize(input_image, input_mask)
  
  #input_image = preprocess_input(input_image)

  return input_image, input_mask
  
train_ds, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=args.dataset_path)
val_ds, test_ds = tfds.load('oxford_iiit_pet:3.*.*', split=['test[:50%]', 'test[-50%:]'], data_dir=args.dataset_path)
print(info)

TRAIN_LENGTH = info.splits['train'].num_examples
BUFFER_SIZE = args.batch_size * 4
STEPS_PER_EPOCH = TRAIN_LENGTH // args.batch_size

train_dataset = (train_ds['train']
                         .map(load_image_train, num_parallel_calls=AUTOTUNE)
                         .cache().shuffle(buffer_size=BUFFER_SIZE).repeat())
validate_dataset = val_ds.map(load_image_test, num_parallel_calls=AUTOTUNE).repeat().batch(args.batch_size)
test_dataset = test_ds.map(load_image_test, num_parallel_calls=AUTOTUNE).batch(args.batch_size)

def augmentedImgGenerator():
  for img, mask in train_dataset:
    img, mask = colorAugmentations(img, mask)
    img, mask = geometryAugmentations(img, mask)
    yield img, mask

def augmentedDataset():
  return (
    tf.data.Dataset.from_generator(
      augmentedImgGenerator, (PIXEL_DATA_TYPE, PIXEL_DATA_TYPE))
       .batch(args.batch_size)
       .prefetch(buffer_size=BUFFER_SIZE))

###############################################################################
# Train the model
###############################################################################
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss',
  mode='min', save_best_only=True, verbose=1)
callbacks = [early_stopper, model_checkpoint]
if args.cyclical_learning_rate:
  callbacks.append(CyclicLR(base_lr=0.0005, max_lr=0.006, step_size=4*STEPS_PER_EPOCH, mode='triangular2'))
  
kernel_size = (3, 3)
model = Sequential()
model.add(BatchNormalization(input_shape=(args.image_size, args.image_size, 3)))
model.add(Conv2D(64, kernel_size, padding='same', strides=1, activation='relu', input_shape=(args.image_size, args.image_size, 3)))
model.add(Conv2D(128, kernel_size, padding='same', strides=1, activation='relu'))
model.add(Conv2D(3, kernel_size, padding='same', strides=1, activation=None))
model.add(keras.layers.Softmax(axis=-1))

model.compile(optimizer='adam',
              loss=CCE,
              metrics=['accuracy'])
              
tf.keras.utils.plot_model(model, show_shapes=True)

VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//args.batch_size//VAL_SUBSPLITS
model_history = model.fit(tfds.as_numpy(augmentedDataset()), epochs=args.max_epochs,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=tfds.as_numpy(validate_dataset),
                          callbacks=callbacks)
                          
###############################################################################
# Load the best model snapshot and evaluate the quality
###############################################################################     
model = load_model(args.model_path, compile=False)     
model.compile(optimizer='adam',
              loss=CCE,
              metrics=['accuracy'])
              
print('Final test set evaluation:')
test_loss, test_accuracy = model.evaluate(tfds.as_numpy(test_dataset), verbose=0, steps=VALIDATION_STEPS//2)
print('Test loss: {:.4f}. Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))

print('Plotting loss')
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

print('Displaying some example predictions from the test set')

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset, num=1):
  for image, mask in dataset.take(num):
    pred_mask = model.predict(image, steps=1)
    display([image[0], mask[0], create_mask(pred_mask)])

show_predictions(test_dataset, 3)
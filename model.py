import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np
import math
import sys

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from tensorflow.data.experimental import AUTOTUNE

# TODO: restore to github version. (https://github.com/qubvel/segmentation_models)
# Currently requires my fork for partial freezing and center dropout
# (https://github.com/AllWashedOut/segmentation_models) 
import segmentation_models as sm
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
parser.add_argument('--encoder_freeze_percent', action='store',
                      default=0.9, dest='encoder_freeze_percent',
                      help='Fraction of encoder layers to set untrainable', type=float)
parser.add_argument('--encoder', action='store',
                      default='mobilenetv2', dest='encoder',
                      help='Encoder backend. See "Backbones" at https://github.com/qubvel/segmentation_models')
parser.add_argument('--image_size', action='store',
                      default=128, dest='image_size',
                      help='Square resize all input images to this many pixels per dimension', type=int)
parser.add_argument('--continue_existing', action='store_true',
                      default=False, dest='continue_existing',
                      help='Load an existing model and continue training')
parser.add_argument('--cyclical_learning_rate', action='store_true',
                      default=False, dest='cyclical_learning_rate',
                      help='Vary the learning rate in a cycle')
parser.add_argument('--tversky', action='store_true',
                      default=False, dest='tversky',
                      help='include tversky loss')
parser.add_argument('--center_dropout', action='store',
                      default=0.0, dest='center_dropout',
                      help='dropout to apply between encoder and decoder', type=float)
                      
args = parser.parse_args()

###############################################################################
# Custom loss
# Sum of Categorical Crossentropy and Tversky losses.
# For more info on Tversky, see https://arxiv.org/abs/1706.05721
###############################################################################
def tversky_loss(y_true, y_pred):
  # Probabilities that a pixel is NOT of the given class
  y_not_pred = K.ones(K.shape(y_true)) - y_pred
  y_not_true = K.ones(K.shape(y_true)) - y_true
  axis = (0, 1, 2)
  numerators = K.sum(y_pred * y_true, axis)
  denominators = numerators + 0.5 * K.sum(y_pred * y_not_true, axis) + 0.5 * K.sum(y_not_pred * y_true, axis)
  smoothing = tf.constant(1e-8)  # prevent some nan results
  score = K.sum((numerators + smoothing)/(denominators + smoothing))
  
  num_classes = K.cast(K.shape(y_true)[-1], 'float32')
  return (num_classes - score) / num_classes

CCE = keras.losses.CategoricalCrossentropy(from_logits=True)
def combined_loss(y_true, y_pred):
  loss = CCE(y_true, y_pred)
  if args.tversky:
    loss += tversky_loss(y_true, y_pred)
  return loss 

if args.continue_existing:
  model = load_model(args.model_path, compile=False)   
else:
  model = sm.Unet(args.encoder, classes=3, encoder_freeze=args.encoder_freeze_percent,
                  activation='relu', input_shape=(args.image_size, args.image_size, 3),
                  center_dropout=args.center_dropout)
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])
preprocess_input = sm.get_preprocessing(args.encoder)

###############################################################################
# Data augmentation
# Vary pixel color/luminance and take random crops.
# Some care is taken to do this at every iteration, rather than only once per
# test run.
###############################################################################
def colorAugmentations(img, mask):
  # Augmentations that alter color but not geometry. That means that they
  # should be applied only the to image, not the mask.
  img = tf.image.random_hue(img, max_delta=0.1)
  img = tf.image.random_saturation(img, lower=0.5, upper=1.3)
  img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
  img = tf.image.random_brightness(img, max_delta=0.3)
  # add noise
  img += tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.03, dtype=tf.float32)
  return img, mask

def geometryAugmentations(img, mask):
  # Augmentations that alter geometry. That means that they
  # should be applied to image AND mask.
  # Some random operations are done on a concatenated image (img & mask)
  # so that the effect is equal on both.
  combined = tf.concat([img, mask], axis=2)
  # Rotation appears to just be too slow to do so frequently.
  # #rotation_max_degrees = 10
  # #combined = tf.keras.preprocessing.image.random_rotation(combined, rotation_max_degrees, fill_mode='reflect')
  # crop out patches, but keep the original aspect ratio. It's tempting to just take
  # square patches here and skip the square resize later. But that prevents the model
  # from learning on images with funky aspect ratios which are common in the test data.
  patch_size = (tf.shape(img)[0] * 9 // 10, tf.shape(img)[1] * 9 // 10)
  combined = tf.image.random_crop(combined, size=[patch_size[0], patch_size[1], 6])  # 3 color channels + 3 mask channel = 6
  combined = tf.image.random_flip_left_right(combined)
  
  img = combined[:, :, :3]
  img.set_shape([patch_size[0], patch_size[1], 3])
  mask = combined[:, :, 3:]
  mask.set_shape([patch_size[0], patch_size[1], 3])
  
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
              loss=combined_loss,
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
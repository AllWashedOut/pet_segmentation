import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np

import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE

import segmentation_models as sm  # https://github.com/qubvel/segmentation_models

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
                      
args = parser.parse_args()

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=args.dataset_path)

IMG_SIZE = 128
BACKBONE = 'mobilenetv2'
# TODO: unfreeze top layers?
model = sm.Unet(BACKBONE, classes=3, encoder_freeze=True, input_shape=(IMG_SIZE, IMG_SIZE, 3))
preprocess_input = sm.get_preprocessing(BACKBONE)
# TODO: try DICE
custom_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=custom_loss,
              metrics=['accuracy'])
              
def colorAugmentations(img, mask):
  # Augmentations that alter color but not geometry. That means that they
  # should be applied only the to image, not the mask.
  #img = tf.image.random_hue(img, max_delta=0.1)
  #img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
  #img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
  #img = tf.image.random_brightness(img, max_delta=0.3)
  # add noise
  #img += tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.03, dtype=tf.float32)
  
  # TODO: enable above and delete below.
  
  with open('foo.txt', 'a') as out_file:
    out_file.write('foo')
  print('foo!!!!!!!!!!!!!!!')
  
  return img, mask

def geometryAugmentations(img, mask):
  # Augmentations that alter geometry. That means that they
  # should be applied to image AND mask.
  seed = random.randint(0, 99999999)
  img = tf.image.random_flip_left_right(img, seed=seed)
  mask = tf.image.random_flip_left_right(mask, seed=seed)
  patch_size = tf.math.minimum(tf.shape(img)[0], tf.shape(img)[1]) * 9 // 10
  img = tf.image.random_crop(img, size=(patch_size, patch_size, 3), seed=seed)
  mask = tf.image.random_crop(mask, size=(patch_size, patch_size, 1), seed=seed)
  # TODO MORE
  
  # TODO: test antialias and resize methods
  #img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), antialias=True, method=tf.image.ResizeMethod.LANCZOS3)
  #mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), antialias=True, method=tf.image.ResizeMethod.LANCZOS3)
  img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
  mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
  
  with open('foo.txt', 'a') as out_file:
    out_file.write('foo')
  print('BAR!!!!!!!!!!!!!!!')
  
  return img, mask

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
  
@tf.function
def load_image_train(datapoint):
  print('Loading image')

  input_image, input_mask = normalize(datapoint['image'], datapoint['segmentation_mask'])
  
  #input_image = preprocess_input(input_image)

  return input_image, input_mask
  
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

  input_image, input_mask = normalize(input_image, input_mask)
  
  #input_image = preprocess_input(input_image)

  return input_image, input_mask
  

TRAIN_LENGTH = info.splits['train'].num_examples
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // args.batch_size

train = (dataset['train'].shuffle(BUFFER_SIZE)
                         .map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .repeat()
                         .map(geometryAugmentations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .map(colorAugmentations, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                         .batch(args.batch_size))
test = dataset['test'].map(load_image_test).repeat().batch(args.batch_size)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if len(display_list[i].shape) == 4:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i][0]))
    else:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
display([sample_image, sample_mask])

# preprocess input
#x_train = preprocess_input(x_train)
#x_val = preprocess_input(x_val)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
  
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image, steps=1)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...], steps=1))])
             
#show_predictions()

early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss',
  mode='min', save_best_only=True, verbose=1)
callbacks = [early_stopper, model_checkpoint]

VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//args.batch_size//VAL_SUBSPLITS
# TODO: augment
model_history = model.fit(tfds.as_numpy(train), epochs=args.max_epochs,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=tfds.as_numpy(test),
                          callbacks=callbacks)
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
                          
model = load_model(args.model_path, compile=False)     
model.compile(optimizer='adam',
              loss=custom_loss,
              metrics=['accuracy'])  

epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

show_predictions(test, 3)
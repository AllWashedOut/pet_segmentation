import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np
import math
import sys

import keras
import efficientnet.keras
from keras.models import load_model
from keras import backend as K
from keras import activations
from tensorflow.data.experimental import AUTOTUNE
from vis.visualization import visualize_activation
from vis.utils import utils
from vis.input_modifiers import Jitter
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', action='store',
                      default='segment.h5', dest='model_path',
                      help='Output path for the saved model')
parser.add_argument('--dataset_path', action='store',
                      default='~/tensorflow_datasets/', dest='dataset_path',
                      help='Path to store input data.')
parser.add_argument('--image_size', action='store',
                      default=128, dest='image_size',
                      help='Square resize all input images to this many pixels per dimension', type=int)
parser.add_argument('--tversky', action='store_true',
                      default=False, dest='tversky',
                      help='include tversky loss')
parser.add_argument('--show_worst_n', action='store',
                      default=5, dest='show_worst_n',
                      help='Show the n worst mask predictions', type=int)
parser.add_argument('--layer_to_visualize', action='store',
                      default=0, dest='layer_to_visualize',
                      help='Visualize max activation of a layer (if non-zero)', type=int)
                      
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

model = load_model(args.model_path, compile=False)   
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])

PIXEL_DATA_TYPE = tf.float32
def normalize(input_image, input_mask, num_classes=3):
  input_image = tf.cast(input_image, PIXEL_DATA_TYPE) / 255.0
  # Onehot encode the class masks, i.e. turn 0 into [1,0,0] and 1 into [0,1,0] and 2 into [0,0,1]
  input_mask = tf.squeeze(tf.one_hot(indices=tf.cast(input_mask, dtype=tf.uint8), depth=num_classes), axis=-2)
  return input_image, input_mask
  
def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (args.image_size, args.image_size))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (args.image_size, args.image_size))

  input_image, input_mask = normalize(input_image, input_mask)
  
  #input_image = preprocess_input(input_image)

  return input_image, input_mask
  
_, test_ds = tfds.load('oxford_iiit_pet:3.*.*', split=['test[:50%]', 'test[-50%:]'], data_dir=args.dataset_path)

test_dataset = test_ds.map(load_image_test, num_parallel_calls=AUTOTUNE).batch(1)
                          
###############################################################################
# Load the best model snapshot and evaluate the quality
###############################################################################     
model = load_model(args.model_path, compile=False)     
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])
              
for i, layer in enumerate(model.layers):
  print('{}: {}'.format(i, layer.name))
              
print('Final test set evaluation:')
test_loss, test_accuracy = model.evaluate(tfds.as_numpy(test_dataset), verbose=0, steps=10)
print('Test loss: {:.4f}. Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))

if args.layer_to_visualize:
  print('Visualizing model activation')
  filters_to_visualize = [0, 1, 2, 3, 4, 5, 6, 7]
                
  for filter_to_visualize in filters_to_visualize:
    print('Visualizing layer {} filter {}'.format(model.layers[args.layer_to_visualize].name, filter_to_visualize))
    visualization = visualize_activation(model, args.layer_to_visualize, filter_indices=filter_to_visualize, input_modifiers=[Jitter(0)])
    plt.imshow(visualization)
    plt.title(f'Filter = {filter_to_visualize}')
    plt.axis('off')
    plt.show()

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
    
def show_worst_predictions(dataset, model, num=1):
  results = []
  for image, mask in dataset:
    pred_mask = model.predict(image, steps=1)
    loss = combined_loss(mask, pred_mask)
    results.append([loss, image, mask, pred_mask])
  results.sort(reverse=True)
  for i in range(num):
    result = results[i]
    img = K.squeeze(result[1], 0)
    mask = K.squeeze(result[2], 0)
    pred_mask = create_mask(result[3])
    print('Worst prediction #{} has loss {}'.format(i, loss))
    display([img, mask, pred_mask])

    print('showing the {} worst mask predictions'.format(args.show_worst_n))
show_worst_predictions(test_dataset, model, num=args.show_worst_n)
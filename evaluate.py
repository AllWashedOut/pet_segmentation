import tensorflow as tf
import argparse

import keras
from keras.models import load_model
from keras import backend as K
from tensorflow.data.experimental import AUTOTUNE

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', action='store',
                      default=64, dest='batch_size',
                      help='Training batch size', type=int)
parser.add_argument('--model_path', action='store',
                      default='segment.h5', dest='model_path',
                      help='Output path for the saved model')
parser.add_argument('--dataset_path', action='store',
                      default='~/tensorflow_datasets/', dest='dataset_path',
                      help='Path to store input data.')
parser.add_argument('--image_size', action='store',
                      default=128, dest='image_size',
                      help='Square resize all input images to this many pixels per dimension', type=int)
                      
args = parser.parse_args()

_, test_ds = tfds.load('oxford_iiit_pet:3.*.*', split=['test[:50%]', 'test[-50%:]'], data_dir=args.dataset_path)
    
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
  numerators = K.sum(y_pred*y_true, axis)
  denominators = numerators + 0.5 * K.sum(y_pred * y_not_true, axis) + 0.5 * K.sum(y_not_pred * y_true, axis)
  smoothing = tf.constant(1e-8)  # prevent some nan results
  score = K.sum((numerators + smoothing)/(denominators + smoothing))
  
  num_classes = K.cast(K.shape(y_true)[-1], 'float32')
  return (num_classes - score) / num_classes

CCE = keras.losses.CategoricalCrossentropy(from_logits=True)
def combined_loss(y_true, y_pred):
  return CCE(y_true, y_pred) + tversky_loss(y_true, y_pred)

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

test_dataset = test_ds.map(load_image_test, num_parallel_calls=AUTOTUNE).batch(args.batch_size)

###############################################################################
# Load the best model snapshot and evaluate the quality
###############################################################################     
model = load_model(args.model_path, compile=False)     
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])
              
print('Final test set evaluation:')
test_loss, test_accuracy = model.evaluate(tfds.as_numpy(test_dataset), verbose=0, steps=2)
print('Test loss: {:.4f}. Test Accuracy: {:.4f}'.format(test_loss, test_accuracy))

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
import tensorflow as tf
import tensorflow_addons as tfa
import argparse
import random
import numpy as np

from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data.experimental import AUTOTUNE

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--patience', action='store',
                      default=20, dest='patience',
                      help='Training patience', type=int)
parser.add_argument('--model_path', action='store',
                      default='segment.h5', dest='model_path',
                      help='Output path for the saved model')
parser.add_argument('--dataset_path', action='store',
                      default='~/tensorflow_datasets/', dest='dataset_path',
                      help='Path to store input data.')
                      
args = parser.parse_args()

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=args.dataset_path)
val_ds, test_ds = tfds.load('oxford_iiit_pet:3.*.*', split=['test[:50%]', 'test[-50%:]'], data_dir=args.dataset_path)

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
  
@tf.function
def load_image(datapoint):
  input_image, input_mask = normalize(datapoint['image'], datapoint['segmentation_mask'])

  return input_image, input_mask

IMAGE_SIZE = 128
#IMAGE_SIZE = 224
@tf.function
def resize(input_image, input_mask):
  input_image = tf.image.resize(input_image, (IMAGE_SIZE, IMAGE_SIZE))
  input_mask = tf.image.resize(input_mask, (IMAGE_SIZE, IMAGE_SIZE))
  return input_image, input_mask
  
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train = dataset['train'].map(load_image, num_parallel_calls=AUTOTUNE)
#train_dataset = dataset['train'].map(load_image).cache().map(augment_image, num_parallel_calls=None).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
validate_dataset = val_ds.map(load_image, num_parallel_calls=AUTOTUNE).map(resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
test_dataset = test_ds.map(load_image, num_parallel_calls=AUTOTUNE).map(resize, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

def colorAugmentations(img):
  # Augmentations that alter color but not geometry. That means that they
  # should be applied only the to image, not the mask.
  img = tf.image.random_hue(img, max_delta=0.1)
  img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
  img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
  img = tf.image.random_brightness(img, max_delta=0.3)
  # add noise
  img += tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.03, dtype=tf.float32)
  
  return img

def generatorFromDataSet(dataset, augment=False):
  seed = random.randint(0, 99999999)
  dataset = dataset.map(resize)  # TODO: do this after warps?
  x = []
  y = []
  for sample in dataset:
    x.append(sample[0].numpy())
    y.append(sample[1].numpy())

  x = np.asarray(x)
  y = np.asarray(y)
  
  # transforms that effect both image and mask
  data_gen_args = dict(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='reflect')
    
  img_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args, preprocessing_function=colorAugmentations)
  mask_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
  img_gen.fit(x, augment=True, seed=seed)
  mask_gen.fit(y, augment=True, seed=seed)
  
  # save_to_dir='.'
  return zip(img_gen.flow(x, seed=seed, batch_size=BATCH_SIZE, shuffle=True),
             mask_gen.flow(y, seed=seed, batch_size=BATCH_SIZE, shuffle=True))

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
for image, mask in train.take(1):
  sample_image, sample_mask = image, mask
#display([sample_image, sample_mask])

OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False)
#base_model = tf.keras.applications.InceptionV3(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False)

#for layer in base_model.layers:
#  print('{}'.format(layer.name))

# Use the activations of these layers
# layer_identifiers = [
    # #288,   # 32x32
    # #86,   # 16x16
    # #16,  # 8x8
    # #9,      # 4x4
    # 'activation_2',
    # 'activation_4'
# ]
layer_identifiers = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(i).output if isinstance(i, str) else base_model.layers[i].output for i in layer_identifiers]
print('Selected encoder activations to pass to the decoder:')
for layer in layers:
  print('\t{}'.format(layer.name))

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

apply_dropout = False
up_stack = [
    pix2pix.upsample(512, 3, apply_dropout=apply_dropout),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3, apply_dropout=apply_dropout),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3, apply_dropout=apply_dropout),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3, apply_dropout=apply_dropout),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
  
def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)
  
@tf.function()
def combined_loss(y_true, y_pred, name='CustomLoss'):
  #return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred) + dice_loss(y_true, y_pred)
  return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
  #return dice_loss(y_true, y_pred)
  
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])
              
tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]
 
def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])
             
#show_predictions()

#class DisplayCallback(tf.keras.callbacks.Callback):
#  def on_epoch_end(self, epoch, logs=None):
#    clear_output(wait=True)
#    show_predictions()
#    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    
# Early stopping with patience
early_stopper = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
model_checkpoint = ModelCheckpoint(args.model_path, monitor='val_loss',
  mode='min', save_best_only=True, verbose=1)
callbacks = [early_stopper, model_checkpoint]
    
MAX_EPOCHS = 200
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
model_history = model.fit(generatorFromDataSet(train, augment=True), epochs=MAX_EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=validate_dataset,
                          callbacks=callbacks)
                          
model = load_model(args.model_path, compile=False)     
model.compile(optimizer='adam',
              loss=combined_loss,
              metrics=['accuracy'])                     
#show_predictions()
                          
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

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

print('Final evaluation:')
model.evaluate(test_dataset)
show_predictions(test_dataset, 3)
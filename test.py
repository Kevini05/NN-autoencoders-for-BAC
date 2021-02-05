# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda, Dropout
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf
from keras import regularizers
from keras.optimizers import Adam
from mish import Mish as mish

import utils_ML
import utils

import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf


def round_sigmoid(x,a):
  return 1 / (1 + tf.exp(-a*(x-0.5)))

def loss_fn(y_true, y_pred):
  binary_neck_loss = tf.abs(0.5 - tf.abs(0.5-y_pred))
  return K.mean(binary_neck_loss, axis=-1)

def gradient_stopper(x):
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  return output

@tf.custom_gradient
def round_function(x):
  output = tf.math.round(x)

  def grad(x):
    return tf.gradients(tf.math.sigmoid(x), x)[0]
  return output, lambda y: grad(x)

@tf.custom_gradient
def BSC_noise(x, epsilon_max):
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """
  two = tf.cast(2, tf.float32)
  x = K.round(x)
  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.) < epsilon_max,tf.float32)
  print("n \n",K.get_value(n))
  y = tf.math.floormod(x+n,two)
  def grad(x):
    return tf.cast(tf.logical_and(x < 1.0, x > 0.0), tf.float32)
  return y, lambda dy: grad(x)

def h2_loss(y_true, y_pred):
  return keras.losses.binary_crossentropy(y_pred,y_pred)

# Test
# tf.random.set_seed(5)
x = tf.linspace(0, 1, 50, name=None, axis=0)
x = tf.cast(x, tf.float32)
print("x \n",K.get_value(x))
# tf.print("x \n",x)

for i in np.linspace(5,200,11):
  with tf.GradientTape() as t:
    t.watch(x)
    # y = BSC_noise(x,0.15)
    y = round_sigmoid(x,i)
    print("y \n",K.get_value(y))

    # tf.print("gradient \n",t.gradient(y, x))
    grad = K.get_value(t.gradient(y, x))
    print("gradient \n",grad)

    # print("sigmoid gradient \n",K.get_value(t.gradient(tf.math.sigmoid(x), x)))

  plt.plot(K.get_value(x),K.get_value(y), label= i)
  # plt.plot(K.get_value(x),grad)
plt.plot(K.get_value(x),K.get_value(x), label= 'identity')

plt.title('Hard sigmoid')
plt.ylabel('output')
plt.xlabel('input')
plt.legend(loc="best")
plt.grid()
plt.show()
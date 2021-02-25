# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf

import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt
from mish import Mish as mish

@tf.function
def pyx(y,codebook, e0, e1):
  """returns the list of A posteriori probabilities P(X/Y) for a
      received codeword y, compared with all elements in the codebook """
  Pyx_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  two = tf.cast(2, tf.float32)
  y=K.round(y)
  for x in codebook:
    d11 = K.sum(x * y)
    d01 = K.sum(tf.math.floormod(x + 1, two) * y)
    d10 = K.sum(x * tf.math.floormod(y + 1, two))
    d00 = K.sum(tf.math.floormod(x + 1, two)  * tf.math.floormod(y + 1, two))
    f = (e0)**d01 * (e1)**d10 * (1 - e1)**d11 * (1 - e0)**d00
    Pyx_list = Pyx_list.write(Pyx_list.size(), f)
  Pyx_list = Pyx_list.stack()
  Pxy_list = Pyx_list/K.sum(Pyx_list)
  return Pxy_list

@tf.function
def map(inputs, e0,e1,k):
  """to be used as a lambda layer, this function computes the A posteriori distribution over the codebook
    for that, this function uses the function 'pyx' the A posteriori distribution for a given codeword y
      :param inputs: is a tensor composed by the x_n and y_n vectors in the communication chain
      the codebook of reference is taken from the first 2^k elements of the tensor x_n (assuming they contains the prediction for all possible messages)
      TRICK: to create lambda layer that allows to iterate over a tensor
          you just need to put the expression @tf.function before the definition of the function"""
  codebook = tf.cast(inputs[0][0:2 ** k], tf.float32)
  soft_map = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  for y in inputs[1]:
    Pxy_map = pyx(y, codebook, e0, e1)
    soft_map = soft_map.write(soft_map.size(), Pxy_map)

  soft_map = soft_map.stack()
  return soft_map


####################################################################################################
########### Neural Network Generator ###################

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  x = Dense(units=256, activation=activation, kernel_initializer=initializer)(inputs_encoder)
  x = BatchNormalization()(x)
  x = Dense(units=128, activation=activation, kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation= 'sigmoid',kernel_initializer=initializer)(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder')
  return model_enc

### Meta model Layers definitions
def meta_model_generator(k,model_enc,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')
  encoded_bits = model_enc(inputs=inputs_meta)
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits
  noisy_bits = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_t, 'epsilon_1_max': train_epsilon_1}, name='noise_layer')(x)
  outLayer = Lambda(map, arguments={'e0': epsilon_t, 'e1': train_epsilon_1, 'k':k}, name='map_layer')([x,noisy_bits])
  meta_model = keras.Model(inputs=inputs_meta, outputs=outLayer, name = 'meta_model')
  return meta_model


# Command line Parameters
N = int(sys.argv[1])
k = int(sys.argv[2])
nb_pkts = int(sys.argv[3])
length_training = sys.argv[4]

# Select training and test length
if length_training == 'medium':
  rep = 128
  epoch_pretrain = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if nb_pkts < 3000 else nb_pkts
elif length_training == 'bug':
  rep = 2
  epoch_pretrain = 10
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 5)),axis=0)
  verbose = 2
  nb_pkts = 10
elif length_training == 'long':
  rep = 256
  epoch_pretrain = 2000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10000 if nb_pkts < 10000 else nb_pkts
else:
  rep = 128
  epoch_pretrain = 100
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]

#Parameters
MAP_test = False
pretrain = False
pretrain_epsilon = 0.05
train_epsilon_1 = 0.0001       #useless for the BSC and epsilon_1 for the BAC

#Training Data set
u_k = np.array(utils.symbols_generator(k))
U_k = np.tile(u_k,(rep,1))
One_hot = np.eye(2 ** k) # List of outputs of NN
One_hot = np.tile(One_hot, (rep, 1)).astype(float)

G,infoBits = utils.polar_generator_matrix(N, k, 'BAC', 0.1)
cn = utils.matrix_codes(u_k, k, G, N)
c = np.array(cn)
c = np.tile(c,(rep,1))

#Hyper parameters
batch_size = 256
initializer = tf.keras.initializers.Orthogonal()
loss = 'mse' #'categorical_crossentropy'  #'kl_divergence'
activation = 'Mish' # Activation function for hidden layers

lr = 0.01
decay = 0.999
# reducing the learning rate by half every 2 epochs
cbks = [LearningRateScheduler(lambda epoch: lr * decay ** epoch)]
optimizer = keras.optimizers.Nadam(lr=lr)
lr_metric = utils_ML.get_lr_metric(optimizer)

# Saved results recovery for plot them later
BER = utils.read_ber_file(N, k, 'BER')
BER = utils.saved_results(BER, N, k)

model_encoder = encoder_generator(N,k)
# pretraining
print("----------------------------------Pretraining to Polar codebook------------------------------------------")
if pretrain:
  ### Compile our models
  model_encoder.compile(loss='mse', optimizer=optimizer, metrics=[utils_ML.ber_metric,lr_metric])
  ### Fit the model
  history = model_encoder.fit(U_k, c, epochs=300, verbose=verbose, shuffle=False, batch_size=batch_size)
  prediction = model_encoder.predict(U_k)
  C = np.round(prediction).astype('int')
  print('***\n Codebook final\n', prediction, '\n', C, "\n***")
  if MAP_test:
    BER['Polar-MAP'] = utils.bit_error_rate(k, cn, nb_pkts, e0, e1, coded=True)
    print(BER['Polar-MAP'])

print("----------------------------------Fine tuning Encoder ------------------------------------------")
meta_model = meta_model_generator(k,model_encoder, False, pretrain_epsilon)
### Compile our models
meta_model.compile(loss=loss, optimizer=optimizer, metrics=[lr_metric])
### Fit the model
history = meta_model.fit(U_k, One_hot, epochs=epoch_pretrain, verbose=verbose, shuffle=False, batch_size=batch_size)

loss_values = history.history['loss']

prediction = model_encoder.predict(U_k)
C = np.round(prediction).astype('int')
print('***\n Codebook final\n',prediction,'\n',C,"\n***")
BER['NN-MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1, coded = True)

#######################Plotting ###################################################################################
# Plot the loss function values calculated during training
fig = plt.figure(figsize=(20,10))
title = f'N={N} k={k} {length_training} - NN Array-MAP-lambda-soft'
plt.semilogy(loss_values  , alpha=0.8 , color='brown',linewidth=0.15)
# Plot the loss function values passed through a filter
filter_size = 100
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown', label=f"BER ($\epsilon_0$ = {pretrain_epsilon})*")

plt.title(f'{title} - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
fig.savefig(f"./figures/LOSS {title}")
#####################################################
# BER plotting
utils.plot_BAC(f'BER {title}', BER, k / N)

print(title)
# plt.show()

# \Python3\python.exe autoencoder_array-MAP-lambda-soft.py 8 4 1000 medium
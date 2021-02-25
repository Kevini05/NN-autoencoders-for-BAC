# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from mish import Mish as mish

import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt


####################################################################################################
########### Neural Network Generator ###################

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  x = Dense(units=512, activation=activation)(inputs_encoder)
  x = BatchNormalization()(x)
  x = Dense(units=256, activation=activation)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder_model')
  return model_enc

### Decoder Layers definitions
def decoder_generator(N,k):
  inputs_decoder = keras.Input(shape=(N,), name='input_decoder')
  x = Dense(units=128, activation=activation)(inputs_decoder)
  x = BatchNormalization()(x)
  x = Dense(units=64, activation=activation)(x)
  x = BatchNormalization()(x)
  outputs_decoder = Dense(units=k, activation='sigmoid')(x)
  return keras.Model(inputs=inputs_decoder, outputs=outputs_decoder, name='decoder_model')

### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')

  encoded_bits = model_enc(inputs=inputs_meta)
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits
  if channel == 'BSC':
    noisy_bits = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_t}, name='noise_layer')(x)
  elif channel == 'BAC':
    noisy_bits = Lambda(utils_ML.BAC_noise,arguments={'epsilon_0_max': epsilon_t, 'epsilon_1_max': train_epsilon_1}, name='noise_layer')(x)
  decoded_bits = model_dec(inputs=noisy_bits)
  ### Model Build
  meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits,name = 'meta_model')
  return meta_model

# Command line Parameters
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
iterations = int(sys.argv[4])
nb_pkts = int(sys.argv[5])
length_training = sys.argv[6]

# Select training and test length
if length_training == 'medium':
  rep = 256
  epoch_pretrain = 600
  epoch_encoder = 300
  epoch_decoder = 1000
  epoch_autocoder = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if nb_pkts < 3000 else nb_pkts
elif length_training == 'bug':
  rep = 256//2**k
  epoch_pretrain = 2
  epoch_encoder = 3
  epoch_decoder = 4
  epoch_autocoder = 5
  e0 = np.linspace(0.001, 1.0, 11)
  verbose = 2
  nb_pkts = 10
elif length_training == 'long':
  rep = 256
  epoch_pretrain = 1000
  epoch_encoder = 300
  epoch_decoder = 1000
  epoch_autocoder = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10000 if nb_pkts < 10000 else nb_pkts
else:
  rep = 128
  epoch_pretrain = 100
  epoch_encoder = 100
  epoch_decoder = 300
  epoch_autocoder = 100
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)), axis=0)
  verbose = 2
  nb_pkts = 1000 if nb_pkts > 1000 else nb_pkts

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]


#Parameters
epsilon_test = [0.01,0.1,0.3,0.55]
loss_weights = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
MAP_test = True
pretraining = True
train_epsilon_1 = 0.002       #useless for the BSC and epsilon_1 for the BAC
pretrain_epsilon = 0.1
encoder_epsilon = 0.1
decoder_epsilon = 0.1

#Training Data set
u_k = utils.symbols_generator(k)
U_k = np.tile(u_k,(rep,1))
G, infoBits = utils.polar_generator_matrix(N,k, channel, 0.1)
c_n = utils.matrix_codes(u_k, k, G, N)
C_n = np.tile(np.array(c_n),(rep,1))

#Hyper parameters
batch_size = 256
initializer = tf.keras.initializers.HeNormal
loss = 'mse' #'categorical_crossentropy'  #'kl_divergence'
activation = 'relu'

lr = 0.001
decay = 0.999
# reducing the learning rate every epoch
cbks = [LearningRateScheduler(lambda epoch: lr * decay ** epoch)]
optimizer = keras.optimizers.Nadam(lr=lr)
lr_metric = utils_ML.get_lr_metric(optimizer)

# Saved results recovery for plot them later
BER = utils.read_ber_file(N, k, 'BER')
BER = utils.saved_results(BER, N, k)
BLER = utils.read_ber_file(N, k, 'BLER')
BLER = utils.saved_results(BLER, N, k, 'BLER')


# pretraining decoder
print("----------------------------------Decoder Pretraining------------------------------------------")
model_decoder = decoder_generator(N,k)
inputs_meta = keras.Input(shape = N)
if channel == 'BSC':
  noisy_bits = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': pretrain_epsilon}, name='noise_layer')(inputs_meta)
elif channel == 'BAC':
  noisy_bits = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': pretrain_epsilon, 'epsilon_1_max': train_epsilon_1}, name='noise_layer')(inputs_meta)
decoded_bits = model_decoder(inputs=noisy_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=decoded_bits)

### Compile our models
meta_model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy',lr_metric])
### Fit the model
history = meta_model.fit(C_n, U_k, epochs=epoch_pretrain, verbose=verbose, shuffle=True, batch_size=batch_size, validation_data=(C_n, U_k), callbacks=cbks)

loss_values = history.history['loss']
accuracy = history.history['binary_accuracy']
val_accuracy = history.history['val_binary_accuracy']

# pretraining encoder
print("----------------------------------Encoder Pretraining------------------------------------------")
model_encoder = encoder_generator(N,k)
meta_model = meta_model_generator(k,channel,model_encoder,model_decoder, True, pretrain_epsilon)

model_encoder = encoder_generator(N,k)
inputs_meta = keras.Input(shape = k, name='input_meta')
encoded_bits = model_encoder(inputs=inputs_meta)
rounded_bits = Lambda(utils_ML.gradient_stopper,name='Rounded')(encoded_bits)
meta_model = keras.Model(inputs=inputs_meta, outputs=rounded_bits)
### Compile our models
meta_model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy',lr_metric])
### Fit the model
history = meta_model.fit(U_k, C_n, epochs=epoch_pretrain, verbose=verbose, shuffle=True, batch_size=batch_size, validation_data=(U_k,C_n), callbacks=cbks)

loss_values += history.history['loss']
accuracy += history.history['binary_accuracy']
val_accuracy += history.history['val_binary_accuracy']

# Fine tunning
for a in range(iterations + 1):
  if a == iterations:  #
    print(f"\n*****************Joint Fine tuning  {a + 1}/{iterations + 1}*********************************\n")
    model_decoder.trainable = True
    model_encoder.trainable = True
    epoch_int = epoch_autocoder
    train_epsilon = encoder_epsilon
    rounding = True
  elif a % 2 == 0:
    print(f"\n*****************Encoder Fine tuning  {a + 1}/{iterations + 1}*********************************\n")
    model_decoder.trainable = False
    model_encoder.trainable = True
    epoch_int = epoch_encoder
    train_epsilon = encoder_epsilon
    rounding = False
  else:
    print(f"\n*****************Decoder Fine tuning  {a + 1}/{iterations + 1}*********************************\n")
    model_decoder.trainable = True
    model_encoder.trainable = False
    epoch_int = epoch_decoder
    train_epsilon = decoder_epsilon
    rounding = True

  meta_model = meta_model_generator(k, channel, model_encoder, model_decoder, False, train_epsilon)
  ### Compile our models
  meta_model.compile(loss=loss, optimizer=optimizer, metrics=['binary_accuracy',lr_metric])

  ### Fit the model
  history = meta_model.fit(U_k, U_k, epochs=epoch_int, verbose=2, shuffle=True, batch_size=batch_size,validation_data=(U_k, U_k), callbacks=cbks)

  loss_values += history.history['loss']
  accuracy += history.history['binary_accuracy']
  val_accuracy += history.history['val_binary_accuracy']

  if a%2==1 or a==iterations:
    C = np.round(model_encoder.predict(u_k)).astype('int')
    print('codebook C is Linear? ', utils.isLinear(C))
    BER[f"auto-array-array_polar_alt-{a//2+1}"], BLER[f"auto-array-array_polar_alt-{a//2+1}"] = utils_ML.bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,model_decoder,'array')


  lr = lr * decay ** epoch_int

if MAP_test:
  C = np.round(model_encoder.predict(u_k)).astype('int')
  BER['NN-MAP'] = utils.bit_error_rate(k, C, nb_pkts//2, e0, e1, coded = True)

#######################Plotting ###################################################################################
title = f'N={N} k={k} {length_training} - NN Array_array_polar_iterative-training'
full_history = {}
full_history['loss'] = loss_values
full_history['accuracy'] = accuracy
full_history['val_accuracy'] = val_accuracy
utils_ML.plot_loss(f'LOSS {title}',full_history)

#####################################################
# BER and BLER plotting
utils.plot_BAC(f'BER {title}', BER, k / N)
utils.plot_BAC(f'BLER {title}', BLER, k / N)

print(title)
# plt.show()


# \Python3\python.exe autoencoder_array-array_polar_alt-training.py BAC 8 4 6 10000 medium
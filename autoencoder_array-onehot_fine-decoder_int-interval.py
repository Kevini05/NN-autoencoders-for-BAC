# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import tensorflow as tf


import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt
from mish import Mish as mish


####################################################################################################
########### Neural Network Generator ###################

### Encoder Layers definitions
def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  x = Dense(units=256, activation=activation, kernel_initializer=initializer)(inputs_encoder)
  x = BatchNormalization()(x)
  x = Dense(units=128, activation=activation, kernel_initializer=initializer)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid', kernel_initializer=initializer)(x)
  ### Model Build
  model_enc = keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder')
  return model_enc

### Decoder Layers definitions
def decoder_generator(N,k):
  # print(k,type(k))
  inputs_decoder = keras.Input(shape = N+4)
  x = Dense(units=S*2**k, activation=activation,name=f"{activation}_decoder")(inputs_decoder)
  x = BatchNormalization()(x)
  outputs_decoder = Dense(units=2**k, activation='softmax',name='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=inputs_decoder, outputs=outputs_decoder, name = 'decoder_model')
  return  model_dec

### Meta model joint training Layers definitions
def meta_model_generator(k,model_enc,model_dec,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')

  encoded_bits = model_enc(inputs=inputs_meta)
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits

  noisy_bits = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0':epsilon_t,'batch_size':batch_size}, name='noise_layer')(x)
  noisy_bits_1 = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0': epsilon_test[0], 'batch_size': batch_size}, name='noise_layer_1')(x)
  noisy_bits_2 = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0': epsilon_test[1], 'batch_size': batch_size}, name='noise_layer_2')(x)
  noisy_bits_3 = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0': epsilon_test[2], 'batch_size': batch_size}, name='noise_layer_3')(x)
  noisy_bits_4 = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0': epsilon_test[3], 'batch_size': batch_size}, name='noise_layer_4')(x)
  noisy_bits_5 = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0': epsilon_test[4], 'batch_size': batch_size}, name='noise_layer_5')(x)

  decoded_bits = model_dec(inputs=noisy_bits)
  decoded_bits_1 = model_dec(inputs=noisy_bits_1)
  decoded_bits_2 = model_dec(inputs=noisy_bits_2)
  decoded_bits_3 = model_dec(inputs=noisy_bits_3)
  decoded_bits_4 = model_dec(inputs=noisy_bits_4)
  decoded_bits_5 = model_dec(inputs=noisy_bits_5)
  ### Model Build
  meta_model = keras.Model(inputs=inputs_meta, outputs=[decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4,decoded_bits_5],name = 'meta_model')
  return meta_model

### Meta model for decoder's training Layers definitions
def meta_dec_model_generator(model_dec,epsilon_t):
  x = keras.Input(shape=N)

  noisy_bits = Lambda(utils_ML.BAC_noise_int_interval, arguments={'epsilon0':epsilon_t,'batch_size':batch_size}, name='noise_layer')(x)
  noisy_bits_1 = Lambda(utils_ML.BAC_noise_int_interval_test, arguments={'epsilon0': epsilon_test[0], 'epsilon_training': epsilon_t}, name='noise_layer_1')(x)
  noisy_bits_2 = Lambda(utils_ML.BAC_noise_int_interval_test, arguments={'epsilon0': epsilon_test[1], 'epsilon_training': epsilon_t}, name='noise_layer_2')(x)
  noisy_bits_3 = Lambda(utils_ML.BAC_noise_int_interval_test, arguments={'epsilon0': epsilon_test[2], 'epsilon_training': epsilon_t}, name='noise_layer_3')(x)
  noisy_bits_4 = Lambda(utils_ML.BAC_noise_int_interval_test, arguments={'epsilon0': epsilon_test[3], 'epsilon_training': epsilon_t}, name='noise_layer_4')(x)
  noisy_bits_5 = Lambda(utils_ML.BAC_noise_int_interval_test, arguments={'epsilon0': epsilon_test[4], 'epsilon_training': epsilon_t}, name='noise_layer_5')(x)

  decoded_bits = model_dec(inputs=noisy_bits)
  decoded_bits_1 = model_dec(inputs=noisy_bits_1)
  decoded_bits_2 = model_dec(inputs=noisy_bits_2)
  decoded_bits_3 = model_dec(inputs=noisy_bits_3)
  decoded_bits_4 = model_dec(inputs=noisy_bits_4)
  decoded_bits_5 = model_dec(inputs=noisy_bits_5)
  ### Model Build
  meta_model = keras.Model(inputs=x, outputs=[decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4,decoded_bits_5],name = 'meta_model')
  return meta_model


# Command line Parameters
N = int(sys.argv[1])
k = int(sys.argv[2])
nb_pkts = int(sys.argv[3])
length_training = sys.argv[4]

# Select training and test length
if length_training == 'medium':
  rep = 256
  epoch_pretrain = 600
  epoch_encoder = 300
  epoch_decoder = 1000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if nb_pkts < 3000 else nb_pkts
elif length_training == 'bug':
  rep = 256//2**k
  epoch_pretrain = 2
  epoch_encoder = 2
  epoch_decoder = 2
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10
elif length_training == 'long':
  rep = 256
  epoch_pretrain = 1000
  epoch_encoder = 300
  epoch_decoder = 1000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10000 if nb_pkts < 10000 else nb_pkts
else:
  rep = 128
  epoch_pretrain = 100
  epoch_encoder = 100
  epoch_decoder = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)), axis=0)
  verbose = 2
  nb_pkts = 1000 if nb_pkts > 1000 else nb_pkts

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]

#Parameters
S = 4
# epsilon_test = [0.001,0.01,0.1,0.3,0.55]
epsilon_test = [0.005,0.025,0.075,0.125,0.5]
loss_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAP_test = True # Flag that allows (or not) the BER over MAP and the NN encoder
pretraining = True
encoder_fine_tuning = False
decoder_fine_tuning = True
train_epsilon_1 = 0.001       #useless for the BSC and epsilon_1 for the BAC
pretrain_epsilon = 0.1
encoder_epsilon = 0.1
decoder_epsilon = 0.1

#Training Data set
u_k = utils.symbols_generator(k)
U_k = np.tile(u_k,(rep,1))
One_hot = np.eye(2 ** k) # List of outputs of NN
One_hot = np.tile(One_hot, (rep, 1))

#Hyper parameters
batch_size = 256
initializer = tf.keras.initializers.Orthogonal()
loss = 'categorical_crossentropy'  #'kl_divergence'
activation = 'Mish'

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


# # pretraining
if pretraining:
  print("----------------------------------Joint Pretraining------------------------------------------")
  model_encoder = encoder_generator(N,k)
  model_decoder = decoder_generator(N,k)
  meta_model = meta_model_generator(k,model_encoder,model_decoder, True, pretrain_epsilon)
  ### Compile our models
  meta_model.compile(loss=loss, optimizer=optimizer,loss_weights=loss_weights)
  ### Fit the model
  history = meta_model.fit(U_k, [One_hot,One_hot,One_hot,One_hot,One_hot,One_hot], epochs=epoch_pretrain, verbose=verbose, shuffle=False, batch_size=batch_size)

  loss_values = history.history['decoder_model_loss']
  loss_values_1 = history.history['decoder_model_1_loss']
  loss_values_2 = history.history['decoder_model_2_loss']
  loss_values_3 = history.history['decoder_model_3_loss']
  loss_values_4 = history.history['decoder_model_4_loss']
  loss_values_5 = history.history['decoder_model_5_loss']

  # C = np.round(model_encoder.predict(u_k)).astype('int')
  # print('codebook C is Linear? ', utils.isLinear(C))
  # BER[f"auto-array-one-int_pretrain"], BLER[f"auto-array-one-int_pretrain"] = utils_ML.bit_error_rate_NN_decoder(N, k, C, nb_pkts, e0, e1,model_decoder, 'one', pretrain_epsilon)


# Fine tuning
lr = lr * decay **epoch_pretrain

if encoder_fine_tuning:
  print("---------------------------------- Encoder Fine Tuning------------------------------------------")
  model_decoder.trainable = False #train encoder
  model_encoder.trainable = True
  rounding = True

  meta_model = meta_model_generator(k, model_encoder, model_decoder, rounding, encoder_epsilon)
  ### Compile our models
  meta_model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)

  ### Fit the model
  history = meta_model.fit(U_k, [One_hot, One_hot, One_hot, One_hot, One_hot, One_hot], epochs=epoch_encoder, verbose=verbose, shuffle=False, batch_size=batch_size)

  loss_values += history.history['decoder_model_loss']
  loss_values_1 += history.history['decoder_model_1_loss']
  loss_values_2 += history.history['decoder_model_2_loss']
  loss_values_3 += history.history['decoder_model_3_loss']
  loss_values_4 += history.history['decoder_model_4_loss']
  loss_values_5 += history.history['decoder_model_5_loss']

if decoder_fine_tuning:
  print("---------------------------------- Decoder Fine Tuning------------------------------------------")

  model_decoder.trainable = True #train decoder
  model_encoder.trainable = False
  optimizer = 'adam'

  meta_dec_model = meta_dec_model_generator(model_decoder,decoder_epsilon)
  ### Compile our models
  meta_dec_model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)

  # Data training set
  c_n = np.round(model_encoder.predict(u_k)).astype('int')
  C_n = np.tile(c_n,(rep,1))
  ### Fit the model
  history = meta_dec_model.fit(C_n, [One_hot, One_hot, One_hot, One_hot, One_hot, One_hot], epochs=epoch_decoder, verbose=verbose, shuffle=False, batch_size=batch_size)

  loss_values += history.history['decoder_model_loss']
  loss_values_1 += history.history['decoder_model_1_loss']
  loss_values_2 += history.history['decoder_model_2_loss']
  loss_values_3 += history.history['decoder_model_3_loss']
  loss_values_4 += history.history['decoder_model_4_loss']
  loss_values_5 += history.history['decoder_model_5_loss']

  C = np.round(model_encoder.predict(u_k)).astype('int')
  print('codebook C is Linear? ', utils.isLinear(C))
  BER[f"auto-array-one-dec"], BLER[f"auto-array-one_dec"] = utils_ML.bit_error_rate_NN_decoder(N, k, C, nb_pkts, e0, e1,model_decoder, 'one', decoder_epsilon)

if MAP_test:
  BER['NN-MAP'] = utils.bit_error_rate(k, C, nb_pkts//2, e0, e1, coded = True)

#######################Plotting ###################################################################################
# Plot the loss function values for the different epsilon, they were calculated during training
fig = plt.figure(figsize=(20,10))
title = f'N={N} k={k} {length_training} - NN Array_Onehot fine decoder int-interval'
plt.semilogy(loss_values  , alpha=0.8 , color='brown',linewidth=0.15)
plt.semilogy(loss_values_1, alpha=0.8, color='blue',linewidth=0.15)
plt.semilogy(loss_values_2, alpha=0.8, color='orange',linewidth=0.15)
plt.semilogy(loss_values_3, alpha=0.8, color='green',linewidth=0.15)
plt.semilogy(loss_values_4, alpha=0.8, color='red',linewidth=0.15)

# Plot the loss function values passed through a filter, it allows to conclude more easily
filter_size = 100
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown', label=f"BER ($\epsilon_0$ = {decoder_epsilon})*")
plt.semilogy(utils_ML.smooth(loss_values_1,filter_size)[filter_size-1:], color='blue', label=f"BER ($\epsilon_0$ = {epsilon_test[0]})")
plt.semilogy(utils_ML.smooth(loss_values_2,filter_size)[filter_size-1:], color='orange', label=f"BER ($\epsilon_0$ = {epsilon_test[1]})")
plt.semilogy(utils_ML.smooth(loss_values_3,filter_size)[filter_size-1:], color='green', label=f"BER ($\epsilon_0$ = {epsilon_test[2]})")
plt.semilogy(utils_ML.smooth(loss_values_4,filter_size)[filter_size-1:], color='red', label=f"BER ($\epsilon_0$ = {epsilon_test[3]})")
plt.semilogy(utils_ML.smooth(loss_values_5,filter_size)[filter_size-1:], color='purple', label=f"BER ($\epsilon_0$ = {epsilon_test[4]})")

plt.title(f'{title} - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
plt.grid()
fig.savefig(f"./figures/LOSS {title}")
#####################################################
# BER and BLER plotting
utils.plot_BAC(f'BER {title}', BER, k / N)
utils.plot_BAC(f'BLER {title}', BLER, k / N)

print(title)
# plt.show()

# \Python3\python.exe autoencoder_array-onehot_fine-decoder_int-interval.py 8 4 1000 medium
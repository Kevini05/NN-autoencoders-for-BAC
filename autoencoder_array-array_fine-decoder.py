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
  x = Dense(units=512, activation=activation, kernel_initializer=initializer, name='first_layer_encoder')(inputs_encoder)
  x = BatchNormalization(name='batch_1')(x)
  x = Dense(units=256, activation=activation, kernel_initializer=initializer, name='second_layer_encoder')(x)
  x = BatchNormalization(name='batch_2')(x)
  outputs_encoder = Dense(units=N, activation='sigmoid', kernel_initializer=initializer, name='output_layer_encoder')(x)
  return keras.Model(inputs=inputs_encoder, outputs=outputs_encoder, name = 'encoder')

### Decoder Layers definitions
def decoder_generator(N,k):
  inputs_decoder = keras.Input(shape=(N,), name='input_decoder')
  x = Dense(units=128, activation=activation, kernel_initializer=initializer, name='first_layer_decoder')(inputs_decoder)
  x = BatchNormalization(name='batch_3')(x)
  x = Dense(units=64, activation=activation, kernel_initializer=initializer, name='second_layer_decoder')(x)
  x = BatchNormalization(name='batch_4')(x)
  outputs_decoder = Dense(units=k, activation='sigmoid', kernel_initializer=initializer, name='output_layer_decoder')(x)
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
    noisy_bits_1 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_test[0]}, name='noise_layer_1')(x)
    noisy_bits_2 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_test[1]}, name='noise_layer_2')(x)
    noisy_bits_3 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_test[2]}, name='noise_layer_3')(x)
    noisy_bits_4 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_test[3]}, name='noise_layer_4')(x)
    noisy_bits_5 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_test[4]}, name='noise_layer_5')(x)
  elif channel == 'BAC':
    noisy_bits = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_t, 'epsilon_1_max': train_epsilon_1}, name='noise_layer')(x)
    noisy_bits_1 = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_test[0], 'epsilon_1_max': train_epsilon_1}, name='noise_layer_1')(x)
    noisy_bits_2 = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_test[1], 'epsilon_1_max': train_epsilon_1}, name='noise_layer_2')(x)
    noisy_bits_3 = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_test[2], 'epsilon_1_max': train_epsilon_1}, name='noise_layer_3')(x)
    noisy_bits_4 = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_test[3], 'epsilon_1_max': train_epsilon_1}, name='noise_layer_4')(x)
    noisy_bits_5 = Lambda(utils_ML.BAC_noise, arguments={'epsilon_0_max': epsilon_test[4], 'epsilon_1_max': train_epsilon_1}, name='noise_layer_5')(x)
  decoded_bits = model_dec(inputs=noisy_bits)
  decoded_bits_1 = model_dec(inputs=noisy_bits_1)
  decoded_bits_2 = model_dec(inputs=noisy_bits_2)
  decoded_bits_3 = model_dec(inputs=noisy_bits_3)
  decoded_bits_4 = model_dec(inputs=noisy_bits_4)
  decoded_bits_5 = model_dec(inputs=noisy_bits_5)
  ### Model Build
  meta_model = keras.Model(inputs=inputs_meta, outputs=[decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4,decoded_bits_5],name = 'meta_model')
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
  rep = 128
  epoch_pretrain = 300
  epoch_encoder = 300
  epoch_decoder = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if nb_pkts < 3000 else nb_pkts
elif length_training == 'bug':
  rep = 1
  epoch_pretrain = 2
  epoch_encoder = 2
  epoch_decoder = 2
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)),axis=0)
  verbose = 2
  nb_pkts = 10
elif length_training == 'long':
  rep = 512
  epoch_pretrain = 2000
  epoch_encoder = 300
  epoch_decoder = 2000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10000 if nb_pkts < 10000 else nb_pkts
else:
  rep = 128
  epoch_pretrain = 100
  epoch_encoder = 100
  epoch_decoder = 100
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)), axis=0)
  nb_pkts = 1000 if nb_pkts < 1000 else nb_pkts
  verbose = 2

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]

#Parameters
epsilon_test = [0.001,0.01,0.1,0.3,0.55]
loss_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
MAP_test = False # Flag that allows (or not) the BER over MAP and the NN encoder
pretraining = True
train_epsilon_1 = 0.002      #useless for the BSC and epsilon_1 for the BAC
pretrain_epsilon = 0.03
encoder_epsilon = 0.03
decoder_epsilon = 0.03

#Training Data set
u_k = utils.symbols_generator(k)
U_k = np.tile(u_k,(rep,1))

#Hyper parameters
batch_size = 256
initializer = tf.keras.initializers.Orthogonal()
loss = 'mse' #'categorical_crossentropy'  #'kl_divergence'
activation = 'Mish' # Activation function for hidden layers

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
  meta_model = meta_model_generator(k,channel,model_encoder,model_decoder, False, pretrain_epsilon)
  ### Compile our models
  meta_model.compile(loss=loss, optimizer=optimizer, metrics=[utils_ML.ber_metric,lr_metric],loss_weights=loss_weights)
  ### Fit the model
  history = meta_model.fit(U_k, [U_k,U_k,U_k,U_k,U_k,U_k], epochs=epoch_pretrain, verbose=verbose, shuffle=False, batch_size=batch_size, callbacks=cbks)

  loss_values = history.history['decoder_model_loss']
  loss_values_1 = history.history['decoder_model_1_loss']
  loss_values_2 = history.history['decoder_model_2_loss']
  loss_values_3 = history.history['decoder_model_3_loss']
  loss_values_4 = history.history['decoder_model_4_loss']
  loss_values_5 = history.history['decoder_model_5_loss']

  metric_values = history.history['decoder_model_ber_metric']
  metric_values_1 = history.history['decoder_model_1_ber_metric']
  metric_values_2 = history.history['decoder_model_2_ber_metric']
  metric_values_3 = history.history['decoder_model_3_ber_metric']
  metric_values_4 = history.history['decoder_model_4_ber_metric']
  metric_values_5 = history.history['decoder_model_5_ber_metric']

  # C = np.round(model_encoder.predict(u_k)).astype('int')
  # print('codebook C is Linear? ', utils.isLinear(C))
  # BER[f"auto-array-array_alt-pretrain"], BLER[f"auto-array-array_alt-pretrain"] = utils_ML.bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,model_decoder,'array')

# Fine tuning

lr = lr * decay ** epoch_pretrain
for a in range(iterations):

  if a%2==1 or a==iterations:
    print(f"\n*****************Fine tuning Decoder {a + 1}/{iterations}*********************************\n")
    model_decoder.trainable = True
    model_encoder.trainable = False
    epoch_int = epoch_decoder
    train_epsilon = decoder_epsilon
    rounding = False
  elif a%2==0:
    print(f"\n*****************Fine tuning Encoder {a + 1}/{iterations}*********************************\n")
    model_decoder.trainable=False
    model_encoder.trainable=True
    epoch_int = epoch_encoder
    train_epsilon = encoder_epsilon
    rounding = False


  ### Compile our models
  meta_model = meta_model_generator(k, channel, model_encoder, model_decoder, rounding, train_epsilon)
  meta_model.compile(loss=loss, optimizer=optimizer, metrics=[utils_ML.ber_metric,lr_metric], loss_weights=loss_weights)

  ### Fit the model
  history = meta_model.fit(U_k, [U_k, U_k, U_k, U_k, U_k, U_k], epochs=epoch_int, verbose=verbose, shuffle=False, batch_size=batch_size)

  loss_values += history.history['decoder_model_loss']
  loss_values_1 += history.history['decoder_model_1_loss']
  loss_values_2 += history.history['decoder_model_2_loss']
  loss_values_3 += history.history['decoder_model_3_loss']
  loss_values_4 += history.history['decoder_model_4_loss']
  loss_values_5 += history.history['decoder_model_5_loss']

  metric_values += history.history['decoder_model_ber_metric']
  metric_values_1 += history.history['decoder_model_1_ber_metric']
  metric_values_2 += history.history['decoder_model_2_ber_metric']
  metric_values_3 += history.history['decoder_model_3_ber_metric']
  metric_values_4 += history.history['decoder_model_4_ber_metric']
  metric_values_5 += history.history['decoder_model_5_ber_metric']

  if a%2==1 or a==iterations:
    C = np.round(model_encoder.predict(u_k)).astype('int')
    print('codebook C is Linear? ', utils.isLinear(C))
    BER[f"auto-array-array_alt-{a//2+1}"], BLER[f"auto-array-array_alt-{a//2+1}"] = utils_ML.bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,model_decoder,'array')

  lr = lr * decay ** epoch_int #useful when use the callback to reduce de learning rate

if MAP_test:
  BER['NN-MAP'] = utils.bit_error_rate(k, C, nb_pkts//2, e0, e1, coded = True)

print("The model is ready to be used...")

# model_decoder.save(f"./autoencoder/model_decoder.h5")
# model_encoder.save(f"./autoencoder/model_encoder.h5")

#######################Plotting ###################################################################################
## Plot the loss function values for the different epsilon, they were calculated during training
fig = plt.figure(figsize=(20,10))
title = f'N={N} k={k} {length_training} - NN Array_array_fine-decoder'
## Commented to not overcharge the loss figure
# plt.semilogy(loss_values  , alpha=0.8 , color='brown',linewidth=0.15)
# plt.semilogy(loss_values_1, alpha=0.8, color='blue',linewidth=0.15)
# plt.semilogy(loss_values_2, alpha=0.8, color='orange',linewidth=0.15)
# plt.semilogy(loss_values_3, alpha=0.8, color='green',linewidth=0.15)
# plt.semilogy(loss_values_4, alpha=0.8, color='red',linewidth=0.15)

# Plot the loss function values passed through a filter, it allows to conclude more easily
filter_size = 100
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown')
plt.semilogy(utils_ML.smooth(loss_values_1,filter_size)[filter_size-1:], color='blue')
plt.semilogy(utils_ML.smooth(loss_values_2,filter_size)[filter_size-1:], color='orange')
plt.semilogy(utils_ML.smooth(loss_values_3,filter_size)[filter_size-1:], color='green')
plt.semilogy(utils_ML.smooth(loss_values_4,filter_size)[filter_size-1:], color='red')
plt.semilogy(utils_ML.smooth(loss_values_5,filter_size)[filter_size-1:], color='purple')

# Plot the BER metric calculated in training
plt.semilogy(metric_values  , linestyle=':', alpha=0.8, color='brown',linewidth=0.45)
plt.semilogy(metric_values_1, linestyle=':', alpha=0.8, color='blue',linewidth=0.45)
plt.semilogy(metric_values_2, linestyle=':', alpha=0.8, color='orange',linewidth=0.45)
plt.semilogy(metric_values_3, linestyle=':', alpha=0.8, color='green',linewidth=0.45)
plt.semilogy(metric_values_4, linestyle=':', alpha=0.8, color='red',linewidth=0.45)
plt.semilogy(metric_values_5, linestyle=':', alpha=0.8, color='purple',linewidth=0.45)

# Plot the BER metric calculated in training passed through a filter
plt.semilogy(utils_ML.smooth(metric_values,filter_size)[filter_size-1:], color='brown', linestyle='--', label=f"BER ($\epsilon_0$ = {decoder_epsilon})*")
plt.semilogy(utils_ML.smooth(metric_values_1,filter_size)[filter_size-1:], color='blue', linestyle='--', label=f"BER ($\epsilon_0$ = {epsilon_test[0]})")
plt.semilogy(utils_ML.smooth(metric_values_2,filter_size)[filter_size-1:], color='orange', linestyle='--', label=f"BER ($\epsilon_0$ = {epsilon_test[1]})")
plt.semilogy(utils_ML.smooth(metric_values_3,filter_size)[filter_size-1:], color='green', linestyle='--', label=f"BER ($\epsilon_0$ = {epsilon_test[2]})")
plt.semilogy(utils_ML.smooth(metric_values_4,filter_size)[filter_size-1:], color='red', linestyle='--', label=f"BER ($\epsilon_0$ = {epsilon_test[3]})")
plt.semilogy(utils_ML.smooth(metric_values_5,filter_size)[filter_size-1:], color='purple', linestyle='--', label=f"BER ($\epsilon_0$ = {epsilon_test[4]})")

# Small function that plots some horizontal lines as the reference values of BER,
if 'Polar(0.1)' in [*BER]:
  l = len(loss_values_1)-1
  Polar = BER['Polar(0.1)']

  list_bklc = []
  if N==16 and k==8:
    keys = [0.001, 0.028000000000000004, 0.01, 0.1, 0.2928571428571429, 0.55]

  else:
    # print("Keys of BKLC: ", [*BKLC])
    keys_bklc = [*Polar]
    absolute_difference_function = lambda list_value: abs(list_value - pretrain_epsilon)
    keys = [ min(keys_bklc, key=absolute_difference_function)]
    for i in epsilon_test:
      absolute_difference_function = lambda list_value: abs(list_value - i)
      keys.append(min(keys_bklc, key=absolute_difference_function))

  for i in keys:
    list_bklc.append(Polar[i][0])

  plt.hlines(list_bklc[0], 0, l, linestyle=':', color='brown', linewidth=1.0)
  plt.hlines(list_bklc[1], 0, l, linestyle=':', color='blue', linewidth=1.0)
  plt.hlines(list_bklc[2], 0, l, linestyle=':', color='orange', linewidth=1.0)
  plt.hlines(list_bklc[3], 0, l, linestyle=':', color='green', linewidth=1.0)
  plt.hlines(list_bklc[4], 0, l, linestyle=':', color='red', linewidth=1.0)
  plt.hlines(list_bklc[5], 0, l, linestyle=':', color='purple', linewidth=1.0)


# plt.semilogy(loss_values, label='Loss')
plt.title(f'{title} - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
plt.grid()
fig.savefig(f"./figures/LOSS {title}")

# Create a dictionary, to introduce the values of the training in order to know the
dict_training = {}
dict_training[epsilon_test[0]] = [utils_ML.smooth(metric_values_1,filter_size)[-1]]
dict_training[epsilon_test[1]] = [utils_ML.smooth(metric_values_2,filter_size)[-1]]
dict_training[pretrain_epsilon] = [utils_ML.smooth(metric_values,filter_size)[-1]]
dict_training[epsilon_test[2]] = [utils_ML.smooth(metric_values_3,filter_size)[-1]]
dict_training[epsilon_test[3]] = [utils_ML.smooth(metric_values_4,filter_size)[-1]]
dict_training[epsilon_test[4]] = [utils_ML.smooth(metric_values_5,filter_size)[-1]]
# BER['Training'] = dict_training

#####################################################
# BER and BLER plotting
utils.plot_BAC(f'BER {title}', BER, k / N)
utils.plot_BAC(f'BLER {title}', BLER, k / N)

print(title)
# plt.show()

# \Python3\python.exe autoencoder_array-array_fine-decoder.py BAC 8 4 0 1000 medium
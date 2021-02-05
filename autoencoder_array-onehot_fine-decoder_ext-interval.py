# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.layers import LeakyReLU
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l1, l2
from mish import Mish as mish

import  ber_bler_calculator as test
import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt


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
  inputs_decoder = keras.Input(shape = N, name='input_decoder')
  inputs_interval = keras.Input(shape=4, name='input_interval_decoder')
  merged_inputs = keras.layers.Concatenate(axis=1,name='merge')([inputs_decoder, inputs_interval])
  x = Dense(units=S*2**k, activation=activation,name=f"{activation}_decoder")(merged_inputs)
  x = BatchNormalization()(x)
  outputs_decoder = Dense(units=2**k, activation='softmax',name='softmax')(x)
  ### Model Build
  model_dec = keras.Model(inputs=[inputs_decoder,inputs_interval], outputs=outputs_decoder, name = 'decoder_model')
  return  model_dec

### Meta model Layers definitions
def meta_model_generator(k,channel,model_enc,model_dec,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')
  inputs_interval = keras.Input(shape=4, name='input_interval_meta')

  encoded_bits = model_enc(inputs=inputs_meta)
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits
  if channel == 'BSC':
    noisy_bits_1 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_t}, name='noise_layer_1')(x)
    noisy_bits_2 = Lambda(utils_ML.BSC_noise, arguments={'epsilon_max': epsilon_t}, name='noise_layer_2')(x)
  elif channel == 'BAC':
    noisy_bits = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_t, 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer')([x, inputs_interval])
    noisy_bits_1 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[0], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_1')([x, inputs_interval])
    noisy_bits_2 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[1], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_2')([x, inputs_interval])
    noisy_bits_3 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[2], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_3')([x, inputs_interval])
    noisy_bits_4 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[3], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_4')([x, inputs_interval])
    noisy_bits_5 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[4], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_5')([x, inputs_interval])
  decoded_bits = model_dec(inputs=[noisy_bits, inputs_interval])
  decoded_bits_1 = model_dec(inputs=[noisy_bits_1, inputs_interval])
  decoded_bits_2 = model_dec(inputs=[noisy_bits_2, inputs_interval])
  decoded_bits_3 = model_dec(inputs=[noisy_bits_3, inputs_interval])
  decoded_bits_4 = model_dec(inputs=[noisy_bits_4, inputs_interval])
  decoded_bits_5 = model_dec(inputs=[noisy_bits_5, inputs_interval])

  meta_model = keras.Model(inputs=[inputs_meta,inputs_interval], outputs=[decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4,decoded_bits_5],name = 'meta_model')
  return meta_model

### Meta model Layers definitions
def meta_dec_model_generator(channel,model_dec,epsilon_t):

  x = keras.Input(shape=N)
  inputs_interval = keras.Input(shape=4, name='input_interval_meta')

  if channel == 'BAC':
    noisy_bits = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_t, 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer')([x, inputs_interval])
    noisy_bits_1 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[0], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_1')([x, inputs_interval])
    noisy_bits_2 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[1], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_2')([x, inputs_interval])
    noisy_bits_3 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[2], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_3')([x, inputs_interval])
    noisy_bits_4 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[3], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_4')([x, inputs_interval])
    noisy_bits_5 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[4], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_5')([x, inputs_interval])
  decoded_bits = model_dec(inputs=[noisy_bits, inputs_interval])
  decoded_bits_1 = model_dec(inputs=[noisy_bits_1, inputs_interval])
  decoded_bits_2 = model_dec(inputs=[noisy_bits_2, inputs_interval])
  decoded_bits_3 = model_dec(inputs=[noisy_bits_3, inputs_interval])
  decoded_bits_4 = model_dec(inputs=[noisy_bits_4, inputs_interval])
  decoded_bits_5 = model_dec(inputs=[noisy_bits_5, inputs_interval])

  meta_model = keras.Model(inputs=[x, inputs_interval], outputs=[decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4,decoded_bits_5],name = 'meta_model')
  return meta_model


#inputs
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
# iterations = int(sys.argv[4])

nb_pkts = int(sys.argv[6])
small_sim = sys.argv[7]

epsilon_test = [0.001,0.01,0.1,0.3,0.55]

S = 4

if small_sim == 'medium':
  rep = 256
  epoch_pretrain = 600
  epoch_encoder = 300
  epoch_decoder = 1000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 3000 if int(sys.argv[6]) < 3000 else int(sys.argv[6])
elif small_sim == 'bug':
  rep = 256//2**k
  epoch_pretrain = 2
  epoch_encoder = 2
  epoch_decoder = 2
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)),axis=0)
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10
elif small_sim == 'long':
  rep = 256
  epoch_pretrain = 1000
  epoch_encoder = 300
  epoch_decoder = 1000
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
  verbose = 2
  nb_pkts = 10000 if int(sys.argv[6]) < 10000 else int(sys.argv[6])
else:
  rep = 128
  epoch_pretrain = 100
  epoch_encoder = 100
  epoch_decoder = 300
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.1, 1, 10)), axis=0)
  verbose = 2
  nb_pkts = 1000 if int(sys.argv[6]) > 1000 else int(sys.argv[6])

e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
e1 = [t for t in e0 if t <= 0.5]

#Parameters
batch_size = 256
MAP_test = True

u_k = utils.symbols_generator(k)
U_k = np.tile(u_k,(rep,1))
Interval = []
idx =[0.7999,0.10,0.08,0.02]
# idx =[0.25,0.25,0.25,0.25] # for proofs of BSC Noise layer
for i in range(4):
  print('elements per interval:', i, round(len(U_k)*idx[i]), len(U_k))
  for j in range(round(len(U_k)*idx[i])):
    Interval.append(np.eye(4)[i].tolist())
Interval = np.reshape(Interval, (len(U_k), 4))
In = np.eye(2**k) # List of outputs of NN
In = np.tile(In,(rep,1))

train_epsilon_1 = 0.001       #useless for the BSC and epsilon_1 for the BAC
pretrain_epsilon = 0.03
encoder_epsilon = 0.03
decoder_epsilon = 0.3
pretraining = True
loss_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

lr = 0.001
decay = 0.99
# reducing the learning rate by half every 2 epochs
cbks = [LearningRateScheduler(lambda epoch: lr * decay ** (epoch // 1))]
#Hyper parameters
initializer = tf.keras.initializers.Orthogonal()
optimizer = keras.optimizers.Nadam(lr=lr)
lr_metric = utils_ML.get_lr_metric(optimizer)
loss = 'categorical_crossentropy' #'categorical_crossentropy'  #'kl_divergence'          # 'mse'
activation = 'Mish'

BER = test.read_ber_file(N, k, 'BER')
BER = test.saved_results(BER, N, k)
BLER = test.read_ber_file(N, k, 'BLER')
# BLER = test.saved_results(BLER, N, k, 'BLER')


# # pretraining
if pretraining:
  print("----------------------------------Pretraining------------------------------------------")
  model_encoder = encoder_generator(N,k)
  model_decoder = decoder_generator(N,k)
  meta_model = meta_model_generator(k,channel,model_encoder,model_decoder, False, pretrain_epsilon)
  ### Compile our models
  model_encoder.compile()
  model_decoder.compile()
  meta_model.compile(loss=loss, optimizer=optimizer,loss_weights=loss_weights, metrics=lr_metric)
  ### Fit the model
  history = meta_model.fit([U_k, Interval], [In,In,In,In,In,In], epochs=epoch_pretrain, verbose=verbose, shuffle=False, batch_size=batch_size)

  loss_values = history.history['decoder_model_loss']
  loss_values_1 = history.history['decoder_model_1_loss']
  loss_values_2 = history.history['decoder_model_2_loss']
  loss_values_3 = history.history['decoder_model_3_loss']
  loss_values_4 = history.history['decoder_model_4_loss']
  loss_values_5 = history.history['decoder_model_5_loss']

  # if len(sys.argv) > 5:
  #   if sys.argv[5] == 'BER':
  #     C = np.round(model_encoder.predict(u_k)).astype('int')
  #     print('codebook C is Linear? ', utils.isLinear(C))
  #     BER[f"auto-array-one-int_pretrain"], BLER[f"auto-array-one-int_pretrain"] = utils_ML.bit_error_rate_NN_interval_dec(N, k, C, nb_pkts, e0, e1,model_decoder,'one',train_epsilon)

# Fine tuning
lr = lr * decay ** (epoch_pretrain // 1)
#
# print("---------------------------------- Encoder Fine Tuning------------------------------------------")
# model_decoder.trainable = True #train encoder
# model_encoder.trainable = True
# epoch_int = epoch_encoder
# train_epsilon_1 = encoder_epsilon ####### Revisar
# train_epsilon = encoder_epsilon
# rounding = True
# ### Compile our models
# meta_model = meta_model_generator(k, channel, model_encoder, model_decoder, rounding, train_epsilon)
#
# model_encoder.compile()
# model_decoder.compile()
# meta_model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)
#
# ### Fit the model
# history = meta_model.fit(U_k, [In, In, In, In, In, In], epochs=epoch_int, verbose=verbose, shuffle=False, batch_size=batch_size)
#
# loss_values += history.history['decoder_model_loss']
# loss_values_1 += history.history['decoder_model_1_loss']
# loss_values_2 += history.history['decoder_model_2_loss']
# loss_values_3 += history.history['decoder_model_3_loss']
# loss_values_4 += history.history['decoder_model_4_loss']
# loss_values_5 += history.history['decoder_model_5_loss']
# # lr = lr * decay ** (epoch_int // 1)

print("---------------------------------- Decoder Fine Tuning------------------------------------------")

model_decoder.trainable = True #train decoder
model_encoder.trainable = False
epoch_int = epoch_decoder
train_epsilon_1 = 0.002
train_epsilon = decoder_epsilon

### Compile our models
meta_dec_model = meta_dec_model_generator(channel,model_decoder,train_epsilon)

model_encoder.compile()
model_decoder.compile()
meta_dec_model.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights, metrics=lr_metric)

c_n = np.round(model_encoder.predict(u_k)).astype('int')
C_n = np.array(c_n)
C_n = np.tile(C_n,(rep,1))
### Fit the model
history = meta_dec_model.fit([C_n, Interval], [In, In, In, In, In, In], epochs=epoch_int, verbose=verbose, shuffle=False, batch_size=batch_size)

loss_values += history.history['decoder_model_loss']
loss_values_1 += history.history['decoder_model_1_loss']
loss_values_2 += history.history['decoder_model_2_loss']
loss_values_3 += history.history['decoder_model_3_loss']
loss_values_4 += history.history['decoder_model_4_loss']
loss_values_5 += history.history['decoder_model_5_loss']

if len(sys.argv) > 5:
  if sys.argv[5] == 'BER':
    C = np.round(model_encoder.predict(u_k)).astype('int')
    print('codebook C is Linear? ', utils.isLinear(C))
    BER[f"auto-array-one-dec"], BLER[f"auto-array-one_dec"] = utils_ML.bit_error_rate_NN_interval_dec(N, k, C, nb_pkts, e0, e1,model_decoder,'one',train_epsilon)

if MAP_test:
  BER['NN-MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1, coded = True)

print("The model is ready to be used...")

##########################################################################################################################



plt.semilogy(loss_values  , alpha=0.8 , color='brown',linewidth=0.15)
plt.semilogy(loss_values_1, alpha=0.8, color='blue',linewidth=0.15)
plt.semilogy(loss_values_2, alpha=0.8, color='orange',linewidth=0.15)
plt.semilogy(loss_values_3, alpha=0.8, color='green',linewidth=0.15)
plt.semilogy(loss_values_4, alpha=0.8, color='red',linewidth=0.15)

filter_size = 30
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown', label=f"BER ($\epsilon_0$ = {decoder_epsilon})*")
plt.semilogy(utils_ML.smooth(loss_values_1,filter_size)[filter_size-1:], color='blue', label=f"BER ($\epsilon_0$ = {epsilon_test[0]})")
plt.semilogy(utils_ML.smooth(loss_values_2,filter_size)[filter_size-1:], color='orange', label=f"BER ($\epsilon_0$ = {epsilon_test[1]})")
plt.semilogy(utils_ML.smooth(loss_values_3,filter_size)[filter_size-1:], color='green', label=f"BER ($\epsilon_0$ = {epsilon_test[2]})")
plt.semilogy(utils_ML.smooth(loss_values_4,filter_size)[filter_size-1:], color='red', label=f"BER ($\epsilon_0$ = {epsilon_test[3]})")
plt.semilogy(utils_ML.smooth(loss_values_5,filter_size)[filter_size-1:], color='purple', label=f"BER ($\epsilon_0$ = {epsilon_test[4]})")

# dict_training = {}
# dict_training[epsilon_test[0]] = [utils_ML.smooth(loss_values_1,filter_size)[-1]]
# dict_training[epsilon_test[1]] = [utils_ML.smooth(loss_values_2,filter_size)[-1]]
# dict_training[pretrain_epsilon] = [utils_ML.smooth(loss_values,filter_size)[-1]]
# dict_training[epsilon_test[2]] = [utils_ML.smooth(loss_values_3,filter_size)[-1]]
# dict_training[epsilon_test[3]] = [utils_ML.smooth(loss_values_4,filter_size)[-1]]
# dict_training[epsilon_test[4]] = [utils_ML.smooth(loss_values_5,filter_size)[-1]]
# BER['Training'] = dict_training

# plt.semilogy(loss_values, label='Loss')
plt.title(f'Array-Onehot-Interval - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
plt.grid()

#####################################################
## TEST
if len(sys.argv) > 5:
  utils.plot_BAC(f'BER N={N} k={k} - ARRAY-ONEHOT-Interval - {nb_pkts} pkts', BER, k / N)
  # utils.plot_BAC(f'BLER N={N} k={k} - ARRAY-ONEHOT', BLER, k / N)
plt.show()


# \Python3\python.exe autoencoder_array-onehot_fine-decoder_ext-interval.py BAC 4 2 2 BER 1000 medium
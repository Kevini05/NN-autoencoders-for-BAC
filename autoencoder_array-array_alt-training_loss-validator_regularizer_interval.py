# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
from keras.layers.core import Dense, Lambda
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from mish import Mish as mish

import utils
import utils_ML

import numpy as np
import matplotlib.pyplot as plt

####################################################################################################
########### Neural Network Generator ###################


def encoder_generator(N,k):
  inputs_encoder = keras.Input(shape=k, name='input_encoder')
  inputs_interval = keras.Input(shape=4, name='input_interval_encoder')
  merged_inputs = keras.layers.Concatenate(axis=1,name='merge')([inputs_encoder, inputs_interval])
  x = Dense(units=128, activation=activation)(merged_inputs)
  x = BatchNormalization()(x)
  x = Dense(units=64, activation=activation)(x)
  x = BatchNormalization()(x)
  outputs_encoder = Dense(units=N, activation='sigmoid')(x)
  ### Model Build
  model_enc = keras.Model(inputs=[inputs_encoder,inputs_interval], outputs=outputs_encoder, name = 'encoder_model')
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
def meta_model_generator(k,model_enc,model_dec,round,epsilon_t):
  inputs_meta = keras.Input(shape=k, name='input_meta')
  inputs_interval = keras.Input(shape=4, name='input_interval_meta')

  encoded_bits = model_enc(inputs=[inputs_meta, inputs_interval])
  if round:
    x = Lambda(utils_ML.gradient_stopper, name='rounding_layer')(encoded_bits)
  else:
    x = encoded_bits

  noisy_bits = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_t, 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer')([x, inputs_interval])
  noisy_bits_1 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[0], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_1')([x, inputs_interval])
  noisy_bits_2 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[1], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_2')([x, inputs_interval])
  noisy_bits_3 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[2], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_3')([x, inputs_interval])
  noisy_bits_4 = Lambda(utils_ML.BAC_noise_interval, arguments={'epsilon0_max': epsilon_test[3], 'epsilon1_max': train_epsilon_1, 'batch_size': batch_size}, name='noise_layer_4')([x, inputs_interval])

  decoded_bits = model_dec(inputs=[noisy_bits, inputs_interval])
  decoded_bits_1 = model_dec(inputs=[noisy_bits_1, inputs_interval])
  decoded_bits_2 = model_dec(inputs=[noisy_bits_2, inputs_interval])
  decoded_bits_3 = model_dec(inputs=[noisy_bits_3, inputs_interval])
  decoded_bits_4 = model_dec(inputs=[noisy_bits_4, inputs_interval])
  # Build Model
  meta_model = keras.Model(inputs=[inputs_meta,inputs_interval], outputs=[encoded_bits,decoded_bits,decoded_bits_1,decoded_bits_2,decoded_bits_3,decoded_bits_4],name = 'meta_model')
  return meta_model

#inputs
# Command line Parameters
N = int(sys.argv[1])
k = int(sys.argv[2])
iterations = int(sys.argv[3])
nb_pkts = int(sys.argv[4])
length_training = sys.argv[5]

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
  epoch_encoder = 2
  epoch_decoder = 2
  epoch_autocoder = 2
  e0 = np.concatenate((np.array([0.001]), np.linspace(0.01, 0.1, 10, endpoint=False), np.linspace(0.1, 1, 15)), axis=0)
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
aux = np.tile(u_k,(rep,N//k))
Interval = []
idx =[0.70,0.15,0.1,0.05]
# idx =[0.25,0.25,0.25,0.25] # for proofs of BSC Noise layer
for i in range(4):
  print('elements per interval:', i, round(len(U_k)*idx[i]))
  for j in range(round(len(U_k)*idx[i])):
    Interval.append(np.eye(4)[i].tolist())
Interval = np.reshape(Interval, (len(U_k), 4))

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



# # pretraining
if pretraining:
  print("----------------------------------Joint Pretraining------------------------------------------")
  model_encoder = encoder_generator(N,k)
  model_decoder = decoder_generator(N,k)
  meta_model = meta_model_generator(k,model_encoder,model_decoder, False, pretrain_epsilon)
  ### Compile our models
  meta_model.compile(loss=[utils_ML.linear_regularizer,loss,loss,loss,loss,loss], optimizer=optimizer, metrics=[lr_metric],loss_weights=loss_weights)
  ### Fit the model
  history = meta_model.fit([U_k, Interval], [aux,U_k,U_k,U_k,U_k,U_k], epochs=epoch_pretrain, verbose=verbose, shuffle=True, batch_size=batch_size)

  loss_list = history.history['loss']
  loss_values = history.history['decoder_model_loss']
  encoder_loss = history.history['encoder_model_loss']
  loss_values_1 = history.history['decoder_model_1_loss']
  loss_values_2 = history.history['decoder_model_2_loss']
  loss_values_3 = history.history['decoder_model_3_loss']
  loss_values_4 = history.history['decoder_model_4_loss']

  # BER[f"auto-array_int_alt-reg"], BLER[f"auto-array_int_alt-reg"] = utils_ML.bit_error_rate_NN_interval(N, k, nb_pkts, e0, e1,model_encoder,model_decoder,'array', train_epsilon)
  lr = lr * decay ** epoch_pretrain
else:
  model_encoder = encoder_generator(N, k)
  model_decoder = decoder_generator(N, k)
  meta_model = meta_model_generator(k, model_encoder, model_decoder, False, pretrain_epsilon)
  loss_list = []
  loss_values = []
  encoder_loss = []
  loss_values_1 = []
  loss_values_2 = []
  loss_values_3 = []
  loss_values_4 = []


# Fine tuning
loss_weights = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
for a in range(iterations + 1):
  print(f"\n*****************Epoch  {a+1}/{iterations}*********************************\n")
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


  meta_model = meta_model_generator(k, model_encoder, model_decoder, rounding, train_epsilon)
  ### Compile our models
  meta_model.compile(loss=[utils_ML.linear_regularizer,loss,loss,loss,loss,loss], optimizer=optimizer, metrics=[lr_metric], loss_weights=loss_weights)

  ### Fit the model
  history = meta_model.fit([U_k, Interval], [aux,U_k, U_k, U_k, U_k, U_k], epochs=epoch_int, verbose=verbose, shuffle=True, batch_size=batch_size, callbacks=cbks)

  loss_list += history.history['loss']
  loss_values += history.history['decoder_model_loss']
  encoder_loss += history.history['encoder_model_loss']
  loss_values_1 += history.history['decoder_model_1_loss']
  loss_values_2 += history.history['decoder_model_2_loss']
  loss_values_3 += history.history['decoder_model_3_loss']
  loss_values_4 += history.history['decoder_model_4_loss']

  if a % 2 == 1 or a == iterations:
    BER[f"auto-array_int_alt-reg-{a//2+1}"], BLER[f"auto-array_int_alt-reg-{a//2+1}"] = utils_ML.bit_error_rate_NN_interval(N, k, nb_pkts, e0, e1, model_encoder, model_decoder,'array',train_epsilon)

  lr = lr * decay ** epoch_pretrain

#######################Plotting ###################################################################################
# Plot the loss function values for the different epsilon, they were calculated during training
fig = plt.figure(figsize=(20,10))
title = f'N={N} k={k} {length_training} - NN Array_array_Iterative_training_regularizer-interval'
plt.semilogy(loss_values,alpha=0.8,color='brown' ,linewidth=0.5, label=f"MSE ($\epsilon_0$ = {decoder_epsilon})")
plt.semilogy(encoder_loss,color='yellow', linewidth=0.5, label=f"Round Loss")
plt.semilogy(loss_values_1, alpha=0.8, color='blue',linewidth=0.15)
plt.semilogy(loss_values_2, alpha=0.8, color='orange',linewidth=0.15)
plt.semilogy(loss_values_3, alpha=0.8, color='green',linewidth=0.15)
plt.semilogy(loss_values_4, alpha=0.8, color='red',linewidth=0.15)

# Plot the loss function values passed through a filter, it allows to conclude more easily
filter_size = 100
plt.semilogy(utils_ML.smooth(loss_values,filter_size)[filter_size-1:], color='brown', label=f'MSE ($\epsilon_0$ = {decoder_epsilon})')
plt.semilogy(utils_ML.smooth(loss_values_1,filter_size)[filter_size-1:], color='blue', label=f"MSE ($\epsilon_0$ = {epsilon_test[0]})")
plt.semilogy(utils_ML.smooth(loss_values_2,filter_size)[filter_size-1:], color='orange', label=f"MSE ($\epsilon_0$ = {epsilon_test[1]})")
plt.semilogy(utils_ML.smooth(loss_values_3,filter_size)[filter_size-1:], color='green', label=f"MSE ($\epsilon_0$ = {epsilon_test[2]})")
plt.semilogy(utils_ML.smooth(loss_values_4,filter_size)[filter_size-1:], color='red', label=f"MSE ($\epsilon_0$ = {epsilon_test[3]})")


plt.title(f'{title} - Training results vs No. epoch - {nb_pkts} pkts')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="best")
plt.grid()
fig.savefig(f"./figures/LOSS {title}")
#####################################################

utils.plot_BAC(f'BER {title}', BER, k / N)
utils.plot_BAC(f'BLER {title}', BLER, k / N)

print(title)
# plt.show()

# \Python3\python.exe autoencoder_array-array_alt-training_loss-validator_regularizer_interval.py 4 2 2 1000 medium

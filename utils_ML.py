# Import deep learning library
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras
import keras.backend as K
import tensorflow as tf


import  ber_bler_calculator as test
import utils

import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

######################## NOISE LAYERS ###################################################
def BSC_noise(x, epsilon_max):
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """
  two = tf.cast(2, tf.float32)
  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.) < epsilon_max,tf.float32)
  y = tf.math.floormod(x+n,two)
  return y # Signal transmis + Bruit


def BAC_noise(x, epsilon_0_max, epsilon_1_max):
  """ parameter : Symboles à envoyer
      return : Symboles bruités + intervale crossover probability"""
  two = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_0_max,tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_1_max,tf.float32)
  n = n0*(x+1) + n1*x
  y = tf.math.floormod(x+n,two) # Signal transmis + Bruit
  return y # Signal transmis + Bruit

def BSC_noise_interval(inputs, epsilon_max , batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles reçus, bruités """
  x = tf.cast(inputs[0],tf.float64)
  interval = inputs[1]
  e = tf.cast(epsilon_max / 4, tf.float64)
  inter = tf.cast(K.argmax(interval), tf.float64) * e + e
  epsilon = K.reshape(tf.cast(inter, tf.float32),shape=(batch_size, 1))

  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.0) < epsilon,tf.float64)
  two = tf.cast(2, tf.float64)
  y = tf.math.floormod(x+n,two)
  return y # Signal transmis + Bruit

def BAC_noise_interval(inputs,  epsilon0_max, epsilon1_max, batch_size):
  """ parameter : Symboles à envoyer
      return : Symboles bruités + intervale crossover probability"""
  x = tf.cast(inputs[0], tf.float64)
  interval = inputs[1]
  e = tf.cast(epsilon0_max/4, tf.float64)
  inter = tf.cast(K.argmax(interval), tf.float64)*e+e
  epsilon0 = K.reshape(tf.cast(inter, tf.float32),shape=(batch_size, 1))

  two = tf.cast(2, tf.float64)

  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0,tf.float64)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1_max,tf.float64)
  n = tf.math.floormod(n0*(x+1) + n1*x, two)
  y = tf.math.floormod(x+n,two) # Signal transmis + Bruit

  return y  # Signal transmis + Bruit

def BAC_noise_int_interval(x, epsilon0, batch_size):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + intervale crossover probability"""

  epsilon1_train_max = 0.002

  epsilon0_train_max = epsilon0
  epsilon0 = np.random.uniform(low=0.0, high=epsilon0_train_max, size=(batch_size, 1))
  epsilon0 = np.reshape(epsilon0, (batch_size, 1))
  epsilon1 = epsilon1_train_max

  interval = np.eye(4)[[int(s * 4 / epsilon0_train_max) for s in epsilon0]]

  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0, tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1, tf.float32)
  n = tf.math.floormod(n0 * (x + 1) + n1 * x, y)

  X = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), y)  # Signal transmis + Bruit
  return tf.concat([X, interval], 1)  # Signal transmis + Bruit + Intervale

def BAC_noise_int_interval_irregular(x, epsilon0, batch_size):
  """ Entrée : Symboles à envoyer
      Sortie : Symboles bruités + interval crossover probability"""

  epsilon1_train_max = 0.002

  epsilon0_train_max = epsilon0
  epsilon0 = np.random.uniform(low=0.0, high=epsilon0_train_max, size=(batch_size, 1))
  # epsilon0 = np.random.chisquare(3, size=(batch_size, 1)) * 0.01
  # epsilon0 = np.random.exponential(0.05, size=(batch_size, 1))
  # epsilon0 = np.random.lognormal(mean=-3.3, sigma=1.8, size=(batch_size, 1))
  # epsilon0 = np.random.gamma(1, 2, size=(batch_size, 1))*0.025

  epsilon0 = np.reshape(epsilon0, (batch_size, 1))
  epsilon1 = epsilon1_train_max

  # interval = np.eye(4)[[int(s * 4 / epsilon0_train_max) for s in epsilon0]]
  interval = np.eye(4)[[int(9.30*s**0.5) if int(9.30*s**0.5)<4 else 3 for s in epsilon0]]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0, tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1, tf.float32)
  n = tf.math.floormod(n0 * (x + 1) + n1 * x, y)

  X = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), y)  # Signal transmis + Bruit
  return tf.concat([X, interval], 1)  # Signal transmis + Bruit + Interval

############################# ROUNDING ####################################################
def gradient_stopper(x):
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  return output

@tf.custom_gradient
def round_function(x):
  output = tf.math.round(x)
  def grad(dy):
    return tf.gradients(1 / (1 + tf.exp(-10*(x-0.5))),x)
  return output, grad

def round_sigmoid(x,a):
  return 1 / (1 + tf.exp(-a*(x-0.5)))

############################### METRICS ####################################################
def get_lr_metric(optimizer):
  def lr(y_true, y_pred):
    return optimizer.lr
  return lr

def ber_metric(y_true, y_pred):
  y_pred = ops.convert_to_tensor_v2(y_pred)
  threshold = math_ops.cast(0.5, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return 1-K.mean(math_ops.equal(y_true, y_pred), axis=-1)

############################### UTILS ######################################################
def smooth(x,filter_size):
  window_len = filter_size
  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
  w = np.hamming(window_len)
  y = np.convolve(w / w.sum(), s, mode='valid')
  return y

def plot_loss(history):

  bler_accuracy = 1 - np.array(history['accuracy'])
  bler_val_accuracy = 1 - np.array(history['val_accuracy'])
  plt.semilogy(bler_accuracy, label='BER - metric (training data)')
  plt.semilogy(bler_val_accuracy, label='BER - metric (validation data)')
  plt.semilogy(history['loss'], label='MSE (training data)')
  plt.title('Training results w.r.t. No. epoch')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.legend(loc="lower left")
  plt.grid()

################################ BER calculators ###########################################
def bit_error_rate_NN(N, k, C, Nb_sequences, e0, e1, model_decoder,output):
  print(f'*******************NN-Decoder******************************************** {Nb_sequences} packets \n')
  print("Decoder Loaded from disk, ready to be used")
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 1
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits
          y_bac = [utils.BAC_channel(xi, ep0, ep1)  for xi in x]# received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          if output == 'one':
            u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict(yh),1) ]  #  NN Detector
          elif output == 'array':
            # print('array',model_decoder.predict(yh))
            u_nn = [idy for idy in np.round(model_decoder.predict(yh)).astype('int').tolist()]  # NN Detector

          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
          # print('u \n', u, 'u_nn \n', u_nn, type(u_nn), 'errors \n', N_errors)
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_interval(N, k, Nb_sequences, e0, e1, model_encoder, model_decoder, output, e_t):
  print(f'*******************NN-Decoder******************************************** {Nb_sequences} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t) if ep0 < e_t else 3] = 1.0
    inter_list = np.array(np.tile(interval, (2 ** k, 1)))
    C = np.round(model_encoder.predict([np.array(U_k), inter_list])).astype('int')

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits

          y_bac = [utils.BAC_channel(xi, ep0, ep1) for xi in x]  # received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          if output == 'one':
            u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict([yh,inter_list]),1) ]  #  NN Detector
          elif output == 'array':
            u_nn = [idy for idy in np.round(model_decoder.predict([yh,inter_list])).astype('int').tolist()]  # NN Detector


          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_interval_dec(N, k, C, Nb_sequences, e0, e1, model_decoder, output, e_t):
  print(f'*******************NN-Decoder******************************************** {Nb_sequences} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t) if ep0 < e_t else 3] = 1.0
    inter_list = np.array(np.tile(interval, (Nb_words, 1)))
    # print(ep0,e_t,interval)

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits

          y_bac = [utils.BAC_channel(xi, ep0, ep1) for xi in x]  # received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          if output == 'one':
            u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict([yh,inter_list]),1) ]  #  NN Detector
          elif output == 'array':
            u_nn = [idy for idy in np.round(model_decoder.predict([yh,inter_list])).astype('int').tolist()]  # NN Detector


          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_decoder(N, k, C, Nb_sequences, e0, e1, model_decoder, output, e_t):
  print(f'*******************NN-Decoder******************************************** {Nb_sequences} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t-0.5) if ep0 < e_t else 3] = 1.0

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits

          y_bac = [utils.BAC_channel(xi, ep0, ep1) for xi in x]  # received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          yh = np.concatenate((yh,inter_list),1)

          if output == 'one':
            u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict(yh),1) ]  #  NN Detector
          elif output == 'array':
            u_nn = [idy for idy in np.round(model_decoder.predict(yh+inter_list)).astype('int').tolist()]  # NN Detector


          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_decoder_irregular(N, k, C, Nb_sequences, e0, e1, model_decoder, output, e_t):
  print(f'*******************NN-Decoder******************************************** {Nb_sequences} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    # interval = np.eye(4)[int(ep0*4/e_t-0.5) if ep0 < e_t else 3]
    interval = np.eye(4)[int(9.30*ep0**0.5) if int(9.30*ep0**0.5)<4 else 3]

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits

          y_bac = [utils.BAC_channel(xi, ep0, ep1) for xi in x]  # received symbols

          yh = np.reshape(y_bac, [Nb_words, N]).astype(np.float64)
          yh = np.concatenate((yh,inter_list),1)

          if output == 'one':
            u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict(yh),1) ]  #  NN Detector
          elif output == 'array':
            u_nn = [idy for idy in np.round(model_decoder.predict(yh+inter_list)).astype('int').tolist()]  # NN Detector


          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def BER_NN(nb_pkts,k,N,model_encoder,model_decoder,e0, MAP_test, input, output):
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]
  if input=='one': #one hot coding
    encoder_input = np.eye(2 ** k)
  elif input == 'array':
    encoder_input = utils.symbols_generator(k)

  C = np.round(model_encoder.predict(encoder_input)).astype('int')
  # print('codebook\n', C)
  print('codebook C is Linear? ', utils.isLinear(C))
  aux = []
  for code in C.tolist():
    if code not in aux:
      aux.append(code)
  nb_repeated_codes = len(C) - len(aux)
  print('+++++++++++++++++++ Repeated Codes NN encoder = ', nb_repeated_codes)
  print('dist = ', sum([sum(codeword) for codeword in C]) * 1.0 / (N * 2 ** k))
  print('***************************************************************')

  if nb_repeated_codes ==0:
    BER = test.read_ber_file(N, k, 'BER')
    BER = test.saved_results(BER, N, k)
    BLER = test.read_ber_file(N, k, 'BLER')
    BLER = test.saved_results(BLER, N, k,'BLER')
    print("NN BER")
    t = time.time()
    BER['auto-non-inter'],BLER['auto-non-inter'] = bit_error_rate_NN(N, k, C, nb_pkts, e0, e1,model_decoder,output)
    t = time.time()-t
    print(f"NN time = {t}s ========================")
    print("BER['auto-NN'] = ", BER['auto-non-inter'])
    print("BLER['auto-NN'] = ", BLER['auto-non-inter'])

    if MAP_test:
      print("MAP BER")
      t = time.time()
      BER['MAP'] = utils.bit_error_rate(k, C, 10000, e0, e1)
      t = time.time()-t
      print(f"MAP time = {t}s =======================")
    utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k} - NN', BER, k / N)
    # utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k} - NN', BLER, k / N)
  else:
    print('Bad codebook repeated codewords')

def BER_NN_interval(nb_pkts,k,N,model_encoder,model_decoder,e0, MAP_test, input, output, train_epsilon):
  e0[len(e0) - 1] = e0[len(e0) - 1] - 0.001
  e1 = [t for t in e0 if t <= 0.5]
  if input=='one': #one hot coding
    encoder_input = np.eye(2 ** k)
  elif input == 'array':
    encoder_input = np.array(utils.symbols_generator(k))

  inter_list = np.array(np.tile([0, 0, 0, 1], (2 ** k, 1)))
  C = np.round(model_encoder.predict([encoder_input,inter_list])).astype('int')
  print('codebook\n', C)
  print('codebook C is Linear? ', utils.isLinear(C))
  aux = []
  for code in C.tolist():
    if code not in aux:
      aux.append(code)
  nb_repeated_codes = len(C) - len(aux)
  print('+++++++++++++++++++ Repeated Codes NN encoder = ', nb_repeated_codes)
  print('dist = ', sum([sum(codeword) for codeword in C]) * 1.0 / (N * 2 ** k))
  print('***************************************************************')

  if nb_repeated_codes ==0:
    BER = test.read_ber_file(N, k, 'BER')
    BER = test.saved_results(BER, N, k)
    BLER = test.read_ber_file(N, k, 'BLER')
    BLER = test.saved_results(BLER, N, k,'BLER')
    print("NN BER")
    t = time.time()
    BER['auto-non-inter'],BLER['auto-non-inter'] = bit_error_rate_NN_interval(N, k, nb_pkts, e0, e1, model_encoder, model_decoder, output, train_epsilon)
    t = time.time()-t
    print(f"NN time = {t}s ========================")
    print("BER['auto-NN'] = ", BER['auto-non-inter'])
    print("BLER['auto-NN'] = ", BLER['auto-non-inter'])

    if MAP_test:
      print("MAP BER")
      t = time.time()
      BER['MAP'] = utils.bit_error_rate(k, C, nb_pkts, e0, e1)
      t = time.time()-t
      print(f"MAP time = {t}s =======================")
    utils.plot_BSC_BAC(f'BER Coding Mechanism N={N} k={k} - NN', BER, k / N)
    # utils.plot_BSC_BAC(f'BLER Coding Mechanism N={N} k={k} - NN', BLER, k / N)
  else:
    print('Bad codebook repeated codewords')

############################################################################


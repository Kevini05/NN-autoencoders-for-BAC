# Import deep learning library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import keras.backend as K
import tensorflow as tf

import utils

import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

######################## NOISE LAYERS ###################################################
def BSC_noise(x, epsilon_max):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel
  @param x: codewords of size N
  @param epsilon_max: training epsilon
  @return: y: noisy codeword
  TRICK: In order to allow the system to find gradients when rounding the input x
    you can use the form tf.stop_gradient(y-x)+x
  """
  x = tf.cast(x, tf.float32)
  two = tf.cast(2, tf.float32)
  n = tf.cast( K.random_uniform(shape=K.shape(x), minval=0.0, maxval=1.) < epsilon_max,tf.float32)
  y = tf.math.floormod(K.round(x)+n,two)
  return tf.stop_gradient(y-x)+x # Transmitted signal + noisy

def BAC_noise(x, epsilon_0_max, epsilon_1_max):
  """ To be used as a lambda layer, this function models the Binary Asymmetric Channel
  @param x: codewords of size N
  @param epsilon_0_max: training epsilon_0
  @param epsilon_1_max: training epsilon_1
  @return: y: noisy codeword
  TRICK: In order to allow the system to find gradients when rounding the input x
    you can use the form tf.stop_gradient(y-x)+x
  """
  x = tf.cast(x, tf.float32)
  two = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_0_max,tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon_1_max,tf.float32)
  n = tf.math.floormod(K.round(n0*(x+1) + n1*x),two)
  y = tf.math.floormod(K.round(x)+n,two) # Signal transmis + Bruit
  # K.print_tensor(x, 'x\n')
  # K.print_tensor(y, 'y\n')
  return tf.stop_gradient(y-x)+x # Transmitted signal + noisy

def BSC_noise_interval(inputs, epsilon_max , batch_size):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses external interval prediction
      uses the function BSC_noise
  @param inputs: It contains two element, [0] codewords of size N and [1] the interval in one hot coding size 4
  @param epsilon_max: training epsilon
  @param batch_size: batch size
  @return: y: noisy codeword
  """
  x = tf.cast(inputs[0],tf.float64)
  interval = inputs[1]
  e = tf.cast(epsilon_max / 4, tf.float64)
  inter = tf.cast(K.argmax(interval), tf.float64) * e + e
  epsilon = K.reshape(tf.cast(inter, tf.float32),shape=(batch_size, 1))

  y = BSC_noise(x, epsilon)
  return y # Transmitted signal + noisy

def BAC_noise_interval(inputs,  epsilon0_max, epsilon1_max, batch_size):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses external interval prediction
      uses the function BAC_noise
  @param inputs: It contains two element, [0] codewords of size N and [1] the interval in one hot coding size 4
  @param epsilon0_max: training epsilon_0
  @param epsilon1_max: training epsilon01
  @param batch_size: batch size
  @return: y: noisy codeword
  """
  x = tf.cast(inputs[0], tf.float64)
  interval = inputs[1]
  e = tf.cast(epsilon0_max/4, tf.float64)
  inter = tf.cast(K.argmax(interval), tf.float64)*e+e
  epsilon0 = K.reshape(tf.cast(inter, tf.float32),shape=(batch_size, 1))

  y = BAC_noise(x, epsilon0, epsilon1_max)
  return y  # Transmitted signal + noisy

def BAC_noise_int_interval(x, epsilon0, batch_size):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses internal interval prediction
      uses the function BAC_noise
  @param x: codewords of size N
  @param epsilon0: training epsilon_0
  @param batch_size: batch size
  @return: y: noisy codeword concatenated with the interval: tensor which is used by the model decoder
  """

  epsilon1_train_max = 0.002

  limits = [epsilon0 / 4 * (i + 1) for i in range(4)]

  idx = np.random.uniform(low=0.0, high=4.0, size=batch_size).astype(int).tolist()
  epsilon0 = [limits[a] if a < 4 else 3 for a in idx]
  epsilon0 = np.reshape(epsilon0, (batch_size, 1))

  interval = np.eye(4)[idx]
  interval = tf.cast(interval, tf.float32)

  y = BAC_noise(x, epsilon0, epsilon1_train_max)
  return tf.concat([y, interval], 1)  # Transmitted signal + noisy + Interval

def BAC_noise_int_interval_test(x, epsilon0, epsilon_training):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses internal interval prediction
       uses the function BAC_noise. Not for training but for test over other values of epsilon
  @param x: codewords of size N
  @param epsilon0: test epsilon_0
  @param epsilon_training: training epsilon_0
  @return: y: noisy codeword concatenated with the interval: tensor which is used by the model decoder
  """

  epsilon1_train_max = 0.002
  batch_size = 256

  interval = np.eye(4)[int(epsilon0*4/epsilon_training) if epsilon0<epsilon_training else 3]
  interval = np.reshape(np.tile(interval, batch_size),(batch_size,4))
  interval = tf.cast(interval, tf.float32)

  y = BAC_noise(x, epsilon0, epsilon1_train_max)
  return tf.concat([y, interval], 1)  # Transmitted signal + noisy + Interval

def BAC_noise_int_interval_irregular(x, epsilon0, batch_size):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses internal irregular interval prediction
      uses the  function BAC_noise
  @param x: codewords of size N
  @param epsilon0: training epsilon_0
  @param batch_size: batch size
  @return: y: noisy codeword concatenated with the interval: tensor which is used by the model decoder
  """

  epsilon1_train_max = 0.002

  limits = np.array([0.1,0.25,0.6,1.0])*epsilon0

  idx = np.random.uniform(low=0.0, high=4.0, size=batch_size).astype(int).tolist()
  epsilon0 = [limits[a] if a < 4 else 3 for a in idx]
  epsilon0 = np.reshape(epsilon0, (batch_size, 1))

  interval = np.eye(4)[idx]
  interval = tf.cast(interval, tf.float32)

  y = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0, tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1_train_max, tf.float32)
  n = tf.math.floormod(n0 * (x + 1) + n1 * x, y)

  X = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), y)  # Signal transmis + Bruit
  return tf.concat([X, interval], 1)  # Transmitted signal + noisy+ Interval

def BAC_noise_int_interval_irregular_test(x, epsilon0, epsilon_training):
  """ To be used as a lambda layer, this function models the Binary Symmetric Channel when uses internal irregular interval prediction
       uses the function BAC_noise. Not for training but for test over other values of epsilon
  @param x: codewords of size N
  @param epsilon0: test epsilon_0
  @param epsilon_training: training epsilon_0
  @return: y: noisy codeword concatenated with the interval: tensor which is used by the model decoder
  """

  epsilon1_train_max = 0.002
  batch_size = 256
  limits = np.array([0.1,0.25,0.6,1.0])*epsilon_training

  interval = np.eye(4)[min(i for i in range(4) if limits[i]-epsilon0>=0) if epsilon0<limits[3] else 3]
  interval = np.reshape(np.tile(interval, batch_size),(batch_size,4))
  interval = tf.cast(interval, tf.float32)


  y = tf.cast(2, tf.float32)
  n0 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon0, tf.float32)
  n1 = tf.cast(K.random_uniform(K.shape(x), minval=0.0, maxval=1.0) < epsilon1_train_max, tf.float32)
  n = tf.math.floormod(n0 * (x + 1) + n1 * x, y)

  X = tf.math.floormod(tf.add(x, tf.cast(n, tf.float32)), y)  # Signal transmis + Bruit
  return tf.concat([X, interval], 1)  # Transmitted signal + noisy + Interval
############################# ROUNDING ####################################################
def gradient_stopper(x):
  """TRICK: In order to allow the system to find gradients when rounding the input x
    you can use the form tf.stop_gradient(x-x)+x"""
  output = tf.stop_gradient(tf.math.round(x)-x)+x
  return output

@tf.custom_gradient
def round_function(x):
  """A possible solution for the rounding problem, here I tried to define a gradient for the function
    different to the default value for the math.roud, but the results weren't better than those
    obtained with the  gradient stopper. However, if you find an appropriate gradient this idea could work well
    TRICK: don't forget to put the expression @tf.custom_gradient to allow you to define a custom gradient"""
  output = tf.math.round(x)
  def grad(dy):
    return tf.gradients(1 / (1 + tf.exp(-10*(x-0.5))),x)
  return output, grad

def round_sigmoid(x,a):
  """Another possible solution for the rounding problem, here I tried to define a diplaced hard-sigmoid, the problem
      with this solution is to find the good value for a, that determines how 'hard' is the sigmoid.
       Bad obtained results as well"""
  return 1 / (1 + tf.exp(-a*(x-0.5)))

############################### METRICS ####################################################
def get_lr_metric(optimizer):
  """ to be used as a metric
  @return the learning rate"""
  def lr(y_true, y_pred):
    return optimizer.lr
  return lr

def ber_metric(y_true, y_pred):
  """ to be used as a metric
    @return the binary error rate, it is exactly the opposite of the binary accuracy"""
  y_pred = ops.convert_to_tensor_v2(y_pred)
  threshold = math_ops.cast(0.5, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return 1-K.mean(math_ops.equal(y_true, y_pred), axis=-1)

############################### UTILS ######################################################
def smooth(x,filter_size):
  """  @return: a smoothed version of the input x  """
  window_len = filter_size
  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
  w = np.hamming(window_len)
  y = np.convolve(w / w.sum(), s, mode='valid')
  return y

def plot_loss(title,history):
  """ plots and saves BER and loss vs the epoch"""
  fig = plt.figure(figsize=(20,10))
  bler_accuracy = 1 - np.array(history['accuracy'])
  bler_val_accuracy = 1 - np.array(history['val_accuracy'])
  plt.semilogy(bler_accuracy, label='BER - metric (training data)')
  plt.semilogy(bler_val_accuracy, label='BER - metric (validation data)')
  plt.semilogy(history['loss'], label='MSE (training data)')
  plt.title(f'{title} - Training results w.r.t. No. epoch')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.legend(loc="lower left")
  plt.grid()
  fig.savefig(f"./figures/{title}")

################################ BER calculators ###########################################
def bit_error_rate_NN(N, k, C, nb_packets, e0, e1, model_decoder, output):
  """ computes the bit an block error rate using the NN-model decoder
  @param C: codebook
  @param nb_packets: number of packets used in the computation
  @param e0 and e1: linspaces containing all the values of epsilon_0 and epsilon_1 to be evaluated
  @param model_decoder: the NN-model of the decoder, previously trained
  @param output: type of output 'array' or 'one' (One-hot coding) it must agree with the type of the decoder model
  @return: two dictionaries 'ber' and 'bler' containing metrics, as keys use the ep0
  """
  print(f'*******************NN-Decoder******************************************** {nb_packets} packets \n')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 1
  Nb_words = int(nb_packets / Nb_iter_max)
  for ep0 in e0:
    ber_row = []
    bler_row = []

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]: #change to this if wants to compute for all epsilon 
      if ep1 == e0[0]:  #just for the most asymmetric case
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
            u_nn = [idy for idy in np.round(model_decoder.predict(yh)).astype('int').tolist()]  # NN Detector

          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * nb_packets)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * nb_packets)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_interval(N, k, nb_packets, e0, e1, model_encoder, model_decoder, output, e_t):
  """ computes the bit an block error rate using the NN-model encoder and decoder when external regular interval
  @param nb_packets: number of packets used in the computation
  @param e0 and e1: linspaces containing all the values of epsilon_0 and epsilon_1 to be evaluated
  @param model_encoder: the NN-model of the encoder, previously trained
  @param model_decoder: the NN-model of the decoder, previously trained
  @param output: type of output 'array' or 'one' (One-hot coding) it must agree with the type of the decoder model
  @param e_t: value of epsilon_0 used during training
  @return: two dictionaries 'ber' and 'bler' containing metrics, as keys use the ep0
  """

  print(f'*******************NN-Decoder******************************************** {nb_packets} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(nb_packets/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t) if ep0 < e_t else 3] = 1.0
    inter_list = np.array(np.tile(interval, (2 ** k, 1)))
    C = np.round(model_encoder.predict([np.array(U_k), inter_list])).astype('int')

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]: #change to this if wants to compute for all epsilon
      if ep1 == e0[0]: #just for the most asymmetric case
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
        ber_row.append(N_errors / (k * 1.0 * nb_packets)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * nb_packets)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_interval_dec(N, k, C, nb_packets, e0, e1, model_decoder, output, e_t):
  """ computes the bit an block error rate using the NN-model decoder when external regular interval
    @param C: codebook
    @param nb_packets: number of packets used in the computation
    @param e0 and e1: linspaces containing all the values of epsilon_0 and epsilon_1 to be evaluated
    @param model_decoder: the NN-model of the decoder, previously trained
    @param output: type of output 'array' or 'one' (One-hot coding) it must agree with the type of the decoder model
    @param e_t: value of epsilon_0 used during training
    @return: two dictionaries 'ber' and 'bler' containing metrics, as keys use the ep0
    """
  print(f'*******************NN-Decoder******************************************** {nb_packets} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(nb_packets/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t) if ep0 < e_t else 3] = 1.0
    inter_list = np.array(np.tile(interval, (Nb_words, 1)))
    # print(ep0,e_t,interval)

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]: #change to this if wants to compute for all epsilon
      if ep1 == e0[0]: #just for the most asymmetric case
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
        ber_row.append(N_errors / (k * 1.0 * nb_packets)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * nb_packets)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_decoder(N, k, C, nb_packets, e0, e1, model_decoder, output, e_t):
  """ computes the bit an block error rate using the NN-model decoder when internal regular interval
      @param C: codebook
      @param nb_packets: number of packets used in the computation
      @param e0 and e1: linspaces containing all the values of epsilon_0 and epsilon_1 to be evaluated
      @param model_decoder: the NN-model of the decoder, previously trained
      @param output: type of output 'array' or 'one' (One-hot coding) it must agree with the type of the decoder model
      @param e_t: value of epsilon_0 used during training
      @return: two dictionaries 'ber' and 'bler' containing metrics, as keys use the ep0
      """
  print(f'*******************NN-Decoder******************************************** {nb_packets} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(nb_packets/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4/e_t-0.5) if ep0 < e_t else 3] = 1.0

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]: #change to this if wants to compute for all epsilon
      if ep1 == e0[0]: #just for the most asymmetric case
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
        ber_row.append(N_errors / (k * 1.0 * nb_packets)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * nb_packets)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def bit_error_rate_NN_decoder_irregular(N, k, C, nb_packets, e0, e1, model_decoder, output):
  """ computes the bit an block error rate using the NN-model decoder when external irregular interval
      @param C: codebook
      @param nb_packets: number of packets used in the computation
      @param e0 and e1: linspaces containing all the values of epsilon_0 and epsilon_1 to be evaluated
      @param model_decoder: the NN-model of the decoder, previously trained
      @param output: type of output 'array' or 'one' (One-hot coding) it must agree with the type of the decoder model
      @return: two dictionaries 'ber' and 'bler' containing metrics, as keys use the ep0
      """
  print(f'*******************NN-Decoder******************************************** {nb_packets} packets')
  U_k = utils.symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(nb_packets/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    # interval = np.eye(4)[int(ep0*4/e_t-0.5) if ep0 < e_t else 3]
    interval = np.eye(4)[int(9.30*ep0**0.5) if int(9.30*ep0**0.5)<4 else 3]

    inter_list = np.array(np.tile(interval, (Nb_words, 1)))

    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]: #change to this if wants to compute for all epsilon
      if ep1 == e0[0]: #just for the most asymmetric case
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits
          # print('uk\n',u,'\nc\n',x)
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
        ber_row.append(N_errors / (k * 1.0 * nb_packets)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * nb_packets)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

############################################################################
# Regularizers
def linear_regularizer(y_true, y_pred):
  """ linear regularizer, this one has the best results in used architectures"""
  binary_neck_loss = 4*tf.abs(0.5 - tf.abs(0.5 -y_pred))
  round_loss = K.mean(binary_neck_loss, axis=-1)
  return round_loss

def crossentropy_regularizer(y_true, y_pred):
  """ regularizer, not really good results in used architectures, I think is because is very restrictive"""
  return tf.losses.binary_crossentropy(y_pred,y_pred)

def quadratic_regularizer(y_true, y_pred):
  """ regularizer, not really good results in used architectures, I think is because is restrictive"""
  return 2*K.mean(y_pred-K.pow(y_pred,2), axis=-1)
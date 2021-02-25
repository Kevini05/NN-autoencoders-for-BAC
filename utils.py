#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import matplotlib.pyplot as plt
import itertools
import keras

from polarcodes import *
import time

def plot_BSC_BAC(title, error_probability,R):
  """
  :param title: Figure title
  :param e0: linspace of all epsilon0
  :param error_probability: error probability dictionary (BER or BLER)
  :param R: Coding rate R=k/N
  :return: plot
  """

  fig = plt.figure(figsize=(7, 3.5), dpi=180, facecolor='w', edgecolor='k')
  fig.subplots_adjust(wspace=0.4, top=0.8)
  fig.suptitle(title, fontsize=14)
  ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
  ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
  marker = itertools.cycle(('h', 'p', '*', '.', '+', 'o', 'h', 's', ','))
  linestyle = itertools.cycle(('-', '--', '-.', ':'))
  legends = []
  for keys in error_probability:
    bac_ber = []
    bsc_ber = []
    legends.append(keys)
    # print(keys,error_probability,e0)
    e0_bac = []
    for ep0 in error_probability[keys]:
      e0_bac.append(ep0)
    for ep0 in e0_bac:
      bac_ber.append(error_probability[keys][ep0][0])
      if ep0 <= 0.5:
        bsc_ber.append(error_probability[keys][ep0][-1])
    # print(keys)
    # print('BAC', ["{:.4f}".format(a) for a in bac_ber])
    # print('BSC', ["{:.4f}".format(a) for a in bsc_ber])

    e0_bsc = [x for x in e0_bac if x <= 0.5]
    m = next(marker)
    # l = next(linestyle)
    l='-'
    ax1.semilogy(e0_bac, [bac_ber[a] for a in range(len(bac_ber))], linestyle=l, marker=m, ms=0.5, linewidth=0.5)
    ax2.semilogy(e0_bsc, [bsc_ber[a] for a in range(len(bsc_ber))], linestyle=l, marker=m, ms=0.5, linewidth=0.5)

  E0 = np.linspace(0.0001, 0.99999, 901)
  ax1.semilogy(E0,cut_off_epsilon(E0, e0_bac[0], R,'BAC'),'k', linestyle='-', ms=0.1, linewidth=0.15)
  E0 = np.linspace(0.0001, 0.49999, 451)
  ax2.semilogy(E0, cut_off_epsilon(E0, 0, R, 'BSC'), 'k', linestyle='-', ms=0.1, linewidth=0.15)

  ax1.legend(legends,prop={'size': 5},loc="lower right")
  ax1.set_title(f"BAC($\epsilon_1$={e0_bac[0]},$\epsilon_0$)", fontsize=8)
  ax1.set_xlabel('$\epsilon_0$', fontsize=8)
  ax1.set_ylabel('Error Probability', fontsize=8)
  # ax1.set_xticklabels(np.arange(0, 1, step=0.2))
  # ax1.grid(which='both', linewidth=0.2)
  ax1.grid(which='major', linewidth=0.2)

  ax2.legend(legends,prop={'size': 5},loc="lower right")
  ax2.set_title('BSC($\epsilon$)', fontsize=8)
  ax2.set_xlabel('$\epsilon$', fontsize=8)
  # ax2.grid(which='both', linewidth=0.2)
  ax2.grid(which='major', linewidth=0.2)

def plot_BAC(title, error_probability,R):
  """
  :param title: Figure title
  :param e0: linspace of all epsilon0
  :param error_probability: error probability dictionary (BER or BLER)
  :param R: Coding rate R=k/N
  :return: plot
  """

  fig = plt.figure(figsize=(7, 3.5), dpi=180, facecolor='w', edgecolor='k')
  fig.subplots_adjust(wspace=0.4, top=0.8)
  fig.suptitle(title, fontsize=14)
  ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
  ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
  legends = [*error_probability]
  for keys in error_probability:
    bac_ber = []
    bsc_ber = []

    e0_bac = [*error_probability[keys]]

    e0_bsc = []
    for ep0 in e0_bac:
      bac_ber.append(error_probability[keys][ep0][0])
      if ep0 <= 0.2:
        bsc_ber.append(error_probability[keys][ep0][0])
        e0_bsc.append(ep0)

    ax1.semilogy(e0_bac, bac_ber, linestyle='-', marker='h', ms=0.5, linewidth=0.5)
    ax2.semilogy(e0_bsc, bsc_ber, linestyle='-', marker='h', ms=0.5, linewidth=0.5)
    # print(e0_bac)


  E0 = np.linspace(0.0001, 0.99999, 901)
  ax1.semilogy(E0,cut_off_epsilon(E0, e0_bac[0], R,'BAC'),'k', linestyle='-', ms=0.1, linewidth=0.15)
  # E0 = np.linspace(0.0001, 0.49999, 451)
  # ax2.semilogy(E0, cut_off_epsilon(E0, 0, R, 'BSC'), 'k', linestyle='-', ms=0.1, linewidth=0.15)

  ax1.legend(legends,prop={'size': 5},loc="lower right")
  ax1.set_title(f"BAC($\epsilon_1$={e0_bac[0]},$\epsilon_0$)", fontsize=8)
  ax1.set_xlabel('$\epsilon_0$', fontsize=8)
  ax1.set_ylabel('Error Probability', fontsize=8)
  # ax1.grid(which='both', linewidth=0.2)
  ax1.grid(which='major', linewidth=0.2)

  ax2.legend(legends,prop={'size': 5},loc="lower right")
  ax2.set_title(f"BAC($\epsilon_1$={e0_bac[0]},$\epsilon_0$)", fontsize=8)
  ax2.set_xlabel('$\epsilon_0$', fontsize=8)
  # ax2.grid(which='both', linewidth=0.2)
  ax2.grid(which='major', linewidth=0.2)
  fig.savefig(f"./figures/{title}")

def h2(x):
  return -(1-x)*np.log2(1-x)-x*np.log2(x)

def cut_off_epsilon(E0,e1,R,channel):
  """
  Returns the cut off epsilon0 for a given e1, and the type of channel (BAC or BSC)
  """
  c = []
  if channel == 'BAC':
    for e0 in E0:
      z = 2**((h2(e0)-h2(e1))/(1-e0-e1))
      c.append(np.log2(z+1) - (1-e1)*h2(e0)/(1-e0-e1) + e0*h2(e1)/(1-e0-e1))
  elif channel == 'BSC':
    for e0 in E0:
      c.append(h2(0.5)-h2(e0))
  index = np.argmin(np.abs(np.array(c) - R))
  cut_off = []
  for i in range(len(E0)):
    cut_off.append(0) if i < index else cut_off.append(0.5)
  return cut_off

def NN_encoder(k,N):
  """
  loads an encoder neural network model (onehot to codeword) and returns its codebook
  """
  print('*******************codebook********************************************')
  one_hot = np.eye(2 ** k)
  model_encoder = keras.models.load_model("trained_models/model_encoder_bsc_16_8_array.h5")
  print("Encoder Loaded from disk, ready to be used")

  codebook = np.round(model_encoder.predict(one_hot)).astype('int')
  # print(codebook)

  return codebook

def block_error_probability(N, k, C, e0, e1):
  """ Computes the block error probability using the error probability equation from a given codebook
      :param N: coded message size
      :param k: message size
      :param C: Codebook
      :return: error probability for all combinations of e0 and e1"""
  U_k = symbols_generator(k)  # all possible messages
  Y_n = symbols_generator(N)  # all possible symbol sequences

  # print("0.00", '|', ["{:.4f}".format(ep1) for ep1 in e1])
  # print('------------------------------------------------------------------')
  error_probability = {}
  for ep0 in e0:
    row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        a = succes_probability(Y_n, C, U_k, ep0, ep1)
        row.append(1 - a)
    error_probability[ep0] = row
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in row])
  return error_probability

def bit_error_rate(k, C, N_iter_max, e0, e1, coded = True):
  """Returns the bit error rate, as a dictionary where the keys are the values of the list e0,
      computing the MAP for a given codebook"""
  print(f"\n *************** BER with the MAP ************** {N_iter_max} packets \n")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:

        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = BAC_channel(x, ep0, ep1)  # received symbols

          te = time.time()
          u_map_bac = U_k[MAP_BAC(y_bac, k, C, ep0, ep1)] if coded else MAP_BAC_uncoded(y_bac, ep0, ep1)  # MAP Detector
          te = time.time() - te
          # print(f"A MAP time = {te}s ========================")

          N_errors += np.sum(np.abs(np.array(u) - np.array(u_map_bac)))  # bit error rate compute with MAPs
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with MAP
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    count += 1
    print("{:.3f}".format(count / len(e0) * 100), '% completed ')
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  return ber

def bit_error_rate_vector(k, C, N_iter_max, e0, e1, coded = True):
  """Returns the bit error rate, as a dictionary where the keys are the values of the list e0,
      computing the MAP for a given codebook"""
  print(f"\n *************** BER with the MAP vector************** {N_iter_max} packets \n")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      # if ep1 == ep0 or ep1 == e0[0]:
      if ep1 == e0[0]:

        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = BAC_channel(x, ep0, ep1)  # received symbols

          te = time.time()
          u_map_bac = U_k[MAP_BAC_vector(y_bac, k, C, ep0, ep1)] if coded else MAP_BAC_uncoded(y_bac, ep0, ep1)  # MAP Detector
          te = time.time() - te
          # print(f"A MAP time = {te}s ========================")

          N_errors += np.sum(np.abs(np.array(u) - np.array(u_map_bac)))  # bit error rate compute with MAPs
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with MAP
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    count += 1
    print("{:.3f}".format(count / len(e0) * 100), '% completed ')
    # print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
  return ber

def bit_error_rate_NN(N, k, C, N_iter_max, e0, e1, channel = 'BSC' ):
  """Loads a decoder neural network model and returns the bit error rate,
        as a dictionary where the keys are the values of the list e0,
        predicting the estimation for a given codebook"""
  print(f"*******************NN-Decoder*********************************  {N_iter_max} packets \n")
  model_decoder = keras.models.load_model("trained_models/model_decoder_bsc_16_8_array.h5")
  print("Decoder Loaded from disk, ready to be used")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  count = 0
  for ep0 in e0:
    ber_row = []
    interval = np.zeros(4)
    interval[int(ep0*4) if ep0 < 0.25 else 3] = 1.0
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        N_errors = 0
        N_iter = 0
        while N_iter < N_iter_max:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1)
          u = U_k[idx]  # Bits to be sent
          x = C[idx]  # coded bits
          y_bac = BAC_channel(x, ep0, ep1)  # received symbols

          yh = np.reshape(np.concatenate((y_bac,interval),axis=0), [1, N+4]) if channel == 'BAC'  else np.reshape(y_bac, [1, N]).astype(np.float64)
          u_nn = U_k[np.argmax(model_decoder(yh))]  #  NN Detector

          N_errors += np.sum(np.abs(np.array(u) - np.array(u_nn)))  # bit error rate compute with NN
        ber_tmp = N_errors / (k * 1.0 * N_iter)  # bit error rate compute with NN
        ber_row.append(ber_tmp)

    ber[ep0] = ber_row
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber

def bit_error_rate_NN_predict(N, k, C, Nb_sequences, e0, e1, inter=False):
  """Loads a decoder neural network model and returns the bit error rate,
          as a dictionary where the keys are the values of the list e0,
          predicting many estimation at the same time for a given codebook"""
  print('*******************NN-Decoder********************************************')
  model_decoder = keras.models.load_model("trained_models/model_decoder_bsc_16_8_array.h5")
  # model_decoder = keras.models.load_model("./model/model_decoder_16_4_std.h5")
  print("Decoder Loaded from disk, ready to be used")
  U_k = symbols_generator(k)  # all possible messages
  ber = {}
  bler = {}
  count = 0
  Nb_iter_max = 10
  Nb_words = int(Nb_sequences/Nb_iter_max)

  for ep0 in e0:
    ber_row = []
    bler_row = []
    interval = np.zeros(4)
    interval[int(ep0*4) if ep0 < 0.25 else 3] = 1.0
    for ep1 in (ep1 for ep1 in e1 if ep1 + ep0 <= 1 and ep1 <= ep0):
      if ep1 == ep0 or ep1 == e0[0]:
        N_errors = 0
        N_errors_bler = 0
        N_iter = 0
        while N_iter < Nb_iter_max:# and N_errors < N_errors_mini:
          N_iter += 1

          idx = np.random.randint(0, len(U_k) - 1, size=(1, Nb_words)).tolist()[0]
          u = [U_k[a] for a in idx]
          x = [C[a] for a in idx]  # coded bits
          if inter:
            y_bac = [np.concatenate((BAC_channel(xi, ep0, ep1), interval), axis=0) for xi in x]  # received symbols
            dec_input_size = N+4
          else:
            y_bac = [BAC_channel(xi, ep0, ep1)  for xi in x]# received symbols
            dec_input_size = N

          yh = np.reshape(y_bac, [Nb_words, dec_input_size]).astype(np.float64)
          u_nn = [U_k[idy] for idy in np.argmax(model_decoder.predict(yh),1) ]  #  NN Detector

          for i in range(len(u)):
            N_errors += np.sum(np.abs(np.array(u[i]) - np.array(u_nn[i])))  # bit error rate compute with NN
            N_errors_bler += np.sum(1.0*(u[i] != u_nn[i]))
        ber_row.append(N_errors / (k * 1.0 * Nb_sequences)) # bit error rate compute with NN
        bler_row.append(N_errors_bler / (1.0 * Nb_sequences)) # block error rate compute with NN

    ber[ep0] = ber_row
    bler[ep0] = bler_row
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in ber_row])
    print("{:.2f}".format(ep0), '|', ["{:.4f}".format(a) for a in bler_row])
    count+= 1
    print("{:.3f}".format(count/len(e0)*100), '% completed ')
  return ber,bler

def mapping(C, X, t, nx):
  """Gallagher's mapping function, the nx-first elements of size t are mapped to '1' """
  codes = []
  count = 0
  if len(C[1]) % t == 0:
    for c in C:
      # print(c)
      row = []
      for i in range(0, len(c), t):
        idx = X.index(c[i:i + t])
        # print(idx)
        row.append(1) if idx < nx else row.append(0)
      count += sum(row)
      codes.append(row)
    print(f"dist = {count * 1.00 / (len(row) * len(codes)):.3f} after mapping")
    aux = []
    a = 0
    for code in codes:
      if code in aux:
        a+=1
    print('++++++++++++++++++Repeated Codes = ', a)
    return codes
  else:
    raise IOError('ERROR t is not multiple of big N')

def mapping2(C, X, t, nx):
  """Gallagher's mapping function, where randomly nx elements of size t are mapped to '1' """
  codes = []
  count = 0
  idx_list = list(range(len(C[1])))
  np.random.shuffle(idx_list)
  # idx_list = [27, 25, 7, 34, 40, 43, 50, 9, 6, 30, 24, 39, 4, 49, 1, 17, 10, 5, 58, 12, 23, 33, 36, 20, 2, 29, 15, 48, 3, 60, 11, 53, 59, 51, 8, 47, 37, 54, 61, 56, 35, 14, 0, 38, 21, 22, 44, 46, 31, 55, 13, 32, 26, 57, 62, 28, 18, 63, 19, 42, 45, 52, 16, 41]
  print(idx_list)
  if len(C[1]) % t == 0:
    for c in C:
      row = []
      # print(c)
      for i in range(0,int(len(C[1])),t):
        aux = [c[a] for a in idx_list[i:i+t]]
        # print(aux)
        idx = X.index(aux)
        # print(idx)
        row.append(1) if idx <= nx else row.append(0)
      count += sum(row)
      codes.append(row)
    print(f"dist = {count * 1.00 / (len(row) * len(codes)):.3f} after mapping")
    aux = []
    for code in codes:
      if code in aux:
        # print('****repeated code******')
        a=1
      else:
        aux.append(code)
    print('+++++++++++++++++++Repeated Codes = ',len(C)-len(aux))
    return codes
  else:
    raise IOError('ERROR t is not multiple of big N')

def integrated_function(infoBits, msm, k, N, threshold):
  """TODO: Not complete
  The first approach to produce the integrated scheme"""
  T = np.transpose(arikan_gen(int(np.log2(N))))
  V = []
  for i in range(len(msm)):
    row = []
    count = 0
    count_frozen = 0
    frozen = [(1 if np.random.randint(0, 100) > threshold else 0) for x in range(N - k)]
    # print(frozen)
    for a in range(N):
      if a in infoBits:
        row.append(msm[i][count])
        count += 1
      else:
        row.append(frozen[count_frozen])
        # row.append(1)
        count_frozen += 1
    V.append(row)
  codebook = matrix_codes2(V, k, T, N)
  # Validation of codewords
  aux = []
  for code in codebook:
    if code in aux:
      print('****repeated codeword Integrated Scheme******')
    else:
      aux.append(code)
  return codebook

################################## BAC Functions #####################################################

def MAP_BAC(symbols,k,codes,e0,e1):
  """ returns the index of the estimated message, computed using the Maximum likelihood
      :param symbols: Received Symbols
      :param k: message size
      :param codes: codebook (all codewords of N-length)
      :param e0 et e1: Crossover probabilities
      :return: index of decoded message among every possible messages """
  g = [0 for i in range(2**k)]
  for j in range(2**k):
    d11 = 0
    d01 = 0
    d10 = 0
    for i in range(len(symbols)):
      d11 += int(codes[j][i]) & int(symbols[i])
      d01 += ~int(codes[j][i]) & int(symbols[i])
      d10 += int(codes[j][i]) & ~int(symbols[i])
    g[j] = (e0/(1-e0))**d01*(e1/(1-e0))**d10*((1-e1)/(1-e0))**d11
  return g.index(max(g))

def MAP_BAC_vector(y,k,codebook,e0,e1):
  """ returns the index of the estimated message, computed using a vector-oriented version Maximum likelihood
      :param y: Received Symbols
      :param k: message size
      :param codebook: codebook (all codewords of N-length)
      :param e0 et e1: Crossover probabilities
      :return: index of decoded message among every possible messages """
  Pyx = []
  print("\n \n*******",e0,e1)
  for x in codebook:
    x = np.array(x)
    d11 = np.sum(x*y)
    d01 = np.sum(np.logical_not(x) * y)
    d10 = np.sum(x * np.logical_not(y))
    d00 = np.sum(np.logical_not(x) *  np.logical_not(y))
    pyx = e0 ** d01 * e1 ** d10 * (1 - e1) ** d11 * (1 - e0) ** d00
    Pyx.append(pyx)
  Pyx = np.array(Pyx)/np.sum(Pyx)
  return np.argmax(Pyx)

def MAP_BAC_uncoded(code,e0,e1):
  """ returns the estimated message for an uncoded communication through the BAC
      :param codes: codebook (all codewords of N-length)
      :param e0 et e1: Crossover probabilities
      :return: index of decoded message among every possible messages """
  if e1+e0==1.0 or e1==0.0 or e0==0:
    y = 0.5
  else:
    y = np.log(e1 / (1 - e0)) / (np.log((e1 * e0) / ((1 - e0) * (1 - e1))))
  decoded_message = []
  for u in code:
    decoded_message.append(1) if u > y else decoded_message.append(0)
  return decoded_message

def symbols_generator(N):
  """ :param N: symbols size (number of bits)
      :return: all possible bit combinations of length N """
  messages = []
  for i in range(2**N):
     messages.append([0 for a in range(N)])
     nb = bin(i)[2:].zfill(N)
     for j in range(N):
        messages[i][j] = int(nb[j])
  return messages

def succes_probability(symbols,codes,msm,e0,e1):
  """ :param symbols: received noisy codewords
      :param
      :param e0 et e1: Crossover probabilities
      :return: the theoretical succes probabilities  """
  Pc = 0
  for y in symbols:
    # print('y',y,'g(y)')
    id = MAP_BAC(y,len(msm[1]),codes,e0,e1)
    u = msm[id]
    d11 = 0
    d01 = 0
    d10 = 0
    for i in range(len(y)):
      d11 += int(codes[id][i]) & int(y[i])
      d01 += ~int(codes[id][i]) & int(y[i])
      d10 += int(codes[id][i]) & ~int(y[i])

    Pc += (e0/(1-e0))**d01*(e1/(1-e0))**d10*((1-e1)/(1-e0))**d11
    # print('u',u,'f(u)',codes[id])
  return (1-e0)**len(y)/(2**len(u))*Pc

def matrix_codes(msm, k, G, N):
  """returns the codebook based on the original messages (msm) and the generator matrix (G)"""
  codes = []
  g = []
  for i in range(N):
    g.append([G[j][i] for j in range(k)])
  # print('G',G,'g',g)
  for a in range(2**k):
    row = [sum([i * j for (i, j) in zip(g[b], msm[a])])%2 for b in range(N)]
    codes.append(row)
  print('dist = ', sum([sum(codes[h]) for h in range(len(codes))])*1.0/(N*2**k))
  return codes

def matrix_codes2(msm, k, G, N):
  """returns the codebook based on the original messages (msm) and the generator matrix (G)"""
  codes = []
  g = []
  for i in range(N):
    g.append([G[j][i] for j in range(N)])
  # print('G',G,'g',g)
  for a in range(2**k):
    row = [sum([i * j for (i, j) in zip(g[b], msm[a])])%2 for b in range(N)]
    codes.append(row)
  print('dist = ', sum([sum(codes[h]) for h in range(len(codes))])*1.000/(N*2**k))
  return codes

def optimal_distribution(e0,e1):
  """returns the capacity achieving distribution for given pair epsilon0 and epsilon1"""

  if e0+e1<1:
    he0	= -e0*np.math.log(e0,2)-(1-e0)*np.math.log(1-e0,2)
    he1 = -e1*np.math.log(e1,2)-(1-e1)*np.math.log(1-e1,2)
    z = 2.0**((he0-he1)/(1.0-e0-e1))
    q = (z-e0*(1+z))/((1+z)*(1-e0-e1))
  else:
    q = 0.5
  return q

def BAC_channel(x, epsilon0, epsilon1):
  """ :param x: codeword to be transmited
      :return: noisy received codeword  """
  x = np.array(x)
  n0 = np.array([int(b0<epsilon0) for b0 in np.random.uniform(0.0, 1.0, len(x))])
  n1 = np.array([int(b1<epsilon1) for b1 in np.random.uniform(0.0, 1.0, len(x))])
  n = n0*(x+1)+n1*x
  return np.mod(n+x,2) # Signal transmis + Bruit

def isLinear(C):
  """returns whether or not the codebook C is linear
      it also computes the minimal Hamming distance and the asymmetric distance of the codebook"""
  isLinear = True
  for i in range(len(C)):
    for j in range(i,len(C)):
      isLinear = list((C[i]+C[j])%2) in C.tolist() and isLinear

  delta = len(C)
  d_h = len(C)
  for i in range(len(C)):
    for j in range(i+1,len(C)):
      x, y = C[i], C[j]
      Nxy = sum([x[l]==0 and y[l]==1 for l in range(len(x))])
      Nyx = sum([y[l] == 0 and x[l] == 1 for l in range(len(x))])
      da = max(Nxy,Nyx)
      delta = min(delta,da)
      d_h = min(d_h,Nxy+Nyx)
  print('Asymmetric distance=', delta,'Hamming Distance=', d_h)
  return isLinear

def saved_results(metric,N=8, k=4, graph = 'BER'):
  if graph == 'BER':
    if N == 16:
      if k == 4:
        a=k
        # metric['BKLC-NN-std'] = {0.001: [0.0], 0.0016378937069540646: [0.0, 0.0], 0.0026826957952797246: [0.0, 0.0], 0.004393970560760791: [0.0, 0.0], 0.0071968567300115215: [0.0, 0.0], 0.011787686347935873: [0.0, 0.0], 0.019306977288832496: [0.0, 5e-05], 0.03162277660168379: [0.0, 0.0003], 0.0517947467923121: [0.0, 0.0014375], 0.08483428982440717: [0.0002125, 0.0079125], 0.13894954943731375: [0.0021625, 0.0409125], 0.22758459260747887: [0.0114875, 0.1598375], 0.3727593720314938: [0.0589, 0.3886125], 0.6105402296585326: [0.227825], 0.999: [0.499325]}
        # metric['BKLC-NN-e0-0.25'] =  {0.001: [0.0], 0.0016378937069540646: [0.0, 0.0], 0.0026826957952797246: [0.0, 0.0], 0.004393970560760791: [0.0, 0.0], 0.0071968567300115215: [0.0, 4.1666666666666665e-05], 0.011787686347935873: [0.0, 5.833333333333333e-05], 0.019306977288832496: [0.0, 0.00013333333333333334], 0.03162277660168379: [0.0, 0.0008666666666666666], 0.0517947467923121: [2.5e-05, 0.00395], 0.08483428982440717: [0.000175, 0.017641666666666667], 0.13894954943731375: [0.0010583333333333334, 0.06320833333333334], 0.22758459260747887: [0.007933333333333334, 0.19488333333333333], 0.3727593720314938: [0.047391666666666665, 0.3892833333333333], 0.6105402296585326: [0.21513333333333334], 0.999: [0.5002416666666667]}
        # metric['auto-NN-softplus'] =  {0.001: [0.0], 0.0016378937069540646: [0.0, 0.0], 0.0026826957952797246: [0.0, 0.0], 0.004393970560760791: [0.0, 2.5e-05], 0.0071968567300115215: [0.0, 0.000275], 0.011787686347935873: [0.0, 0.0009], 0.019306977288832496: [0.0, 0.0014], 0.03162277660168379: [2.5e-05, 0.003925], 0.0517947467923121: [5e-05, 0.009275], 0.08483428982440717: [5e-05, 0.029875], 0.13894954943731375: [0.00095, 0.078775], 0.22758459260747887: [0.004925, 0.20645], 0.3727593720314938: [0.02955, 0.41255], 0.6105402296585326: [0.1618], 0.999: [0.51225]}
        # metric['auton-NN-BSC-non-inter_array'] =  {0.001: [0.0], 0.01: [3.25e-05, 6.25e-05], 0.019000000000000003: [7e-05, 0.0003575], 0.028000000000000004: [0.0001575, 0.0007225], 0.037000000000000005: [0.0002575, 0.0013175], 0.046000000000000006: [0.0005, 0.0023875], 0.05500000000000001: [0.0008725, 0.003745], 0.064: [0.0011625, 0.00501], 0.073: [0.0015025, 0.0076225], 0.082: [0.002055, 0.0104525], 0.09100000000000001: [0.0024925, 0.0136025], 0.1: [0.0032075, 0.0177475], 0.1642857142857143: [0.0133175, 0.065305], 0.2285714285714286: [0.0321875, 0.14786], 0.2928571428571429: [0.061075, 0.24845], 0.3571428571428572: [0.102035, 0.348605], 0.4214285714285715: [0.148445, 0.43118], 0.48571428571428577: [0.1980925, 0.4893675], 0.55: [0.25004], 0.6142857142857143: [0.29828], 0.6785714285714286: [0.3401175], 0.7428571428571429: [0.379995], 0.8071428571428572: [0.4156775], 0.8714285714285716: [0.4453025], 0.9357142857142858: [0.4750925], 0.999: [0.50031]}
        # metric['auton-NN-BAC-non-inter_array'] =  {0.001: [0.0], 0.020900000000000002: [7.5e-06, 0.0004475], 0.0408: [0.000165, 0.0019], 0.060700000000000004: [0.0002175, 0.0056775], 0.0806: [0.0010725, 0.01526], 0.1005: [0.00207, 0.0263225], 0.12040000000000001: [0.0030875, 0.042355], 0.1403: [0.004795, 0.0596275], 0.1602: [0.0073775, 0.08145], 0.1801: [0.00974, 0.10601], 0.2: [0.0135325, 0.1338025], 0.3142857142857143: [0.04972, 0.3042925], 0.4285714285714286: [0.115515, 0.4437675], 0.5428571428571429: [0.2028075], 0.6571428571428573: [0.300325], 0.7714285714285716: [0.38705], 0.8857142857142857: [0.4511825], 0.999: [0.499495]}
        # metric['dec-NN-no-interval'] =  {0.001: [0.0], 0.01: [0.0], 0.019000000000000003: [0.0], 0.028000000000000004: [0.0], 0.037000000000000005: [0.0], 0.046000000000000006: [3.75e-05], 0.05500000000000001: [0.0001625], 0.064: [0.00025], 0.073: [0.00055], 0.082: [0.0005375], 0.09100000000000001: [0.00095], 0.1: [0.0012], 0.1642857142857143: [0.0082875], 0.2285714285714286: [0.02235], 0.2928571428571429: [0.0471125], 0.3571428571428572: [0.086975], 0.4214285714285715: [0.1347875], 0.48571428571428577: [0.1969125], 0.55: [0.2622125], 0.6142857142857143: [0.3314], 0.6785714285714286: [0.3929625], 0.7428571428571429: [0.434575], 0.8071428571428572: [0.46615], 0.8714285714285716: [0.482625], 0.9357142857142858: [0.482225], 0.999: [0.4839]}
        # metric['dec-NN-interval'] =  {0.001: [0.0], 0.01: [0.0], 0.019000000000000003: [0.0], 0.028000000000000004: [0.0], 0.037000000000000005: [3.75e-05], 0.046000000000000006: [3.75e-05], 0.05500000000000001: [1.25e-05], 0.064: [1.25e-05], 0.073: [7.5e-05], 0.082: [0.0001375], 0.09100000000000001: [0.0002625], 0.1: [0.0003875], 0.1642857142857143: [0.002025], 0.2285714285714286: [0.0069], 0.2928571428571429: [0.0180125], 0.3571428571428572: [0.0412375], 0.4214285714285715: [0.0707375], 0.48571428571428577: [0.114725], 0.55: [0.1753625], 0.6142857142857143: [0.2344125], 0.6785714285714286: [0.3104625], 0.7428571428571429: [0.3753625], 0.8071428571428572: [0.425775], 0.8714285714285716: [0.4670625], 0.9357142857142858: [0.481575], 0.999: [0.4839375]}
      elif k == 8:
        a=k
        # metric['Polar(0.1)'] = {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.00025], 0.007943282347242814: [0.000625, 0.0], 0.015848931924611134: [0.000625, 0.006625], 0.03162277660168379: [0.00625, 0.022875], 0.0630957344480193: [0.01375, 0.07925], 0.12589254117941676: [0.0475, 0.18], 0.25118864315095796: [0.13975, 0.3805], 0.501187233627272: [0.315375], 0.999: [0.502125]}
        # metric['BCH(0)'] = {0.001: [0.000375], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.00025], 0.007943282347242814: [0.000125, 0.001375], 0.015848931924611134: [0.000625, 0.003625], 0.03162277660168379: [0.002125, 0.011875], 0.0630957344480193:[0.005375, 0.045875], 0.12589254117941676: [0.028375, 0.141875], 0.25118864315095796: [0.092375, 0.29925], 0.501187233627272: [0.268], 0.999: [0.499875]}
        # metric['BKLC'] = {0.001: [0.0], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [0.0, 0.0], 0.015848931924611134: [0.0, 0.000375], 0.03162277660168379: [0.0, 0.004125], 0.0630957344480193: [0.00225, 0.024875], 0.12589254117941676: [0.011, 0.104], 0.25118864315095796: [0.05775, 0.290375], 0.501187233627272: [0.234], 0.999: [0.50075]}
        # metric['BKLC-NN-std'] ={0.001: [0.0], 0.1009: [0.0094125, 0.0644], 0.2008: [0.0505, 0.223], 0.3007: [0.1182125, 0.34436875], 0.4006: [0.1953375, 0.4312], 0.5005: [0.2701], 0.6004: [0.3314], 0.7003: [0.38294375], 0.8002: [0.4245], 0.9001: [0.46371875], 0.999: [0.4988375]}
        # metric['Polar'] = {0.001: [0.0], 0.0408: [0.006475, 0.0337625], 0.0806: [0.0191625, 0.0964875], 0.12040000000000001: [0.0393875, 0.1788875], 0.1602: [0.062425, 0.2447375], 0.2: [0.0914875, 0.3156125], 0.3142857142857143: [0.177725, 0.435875], 0.4285714285714286: [0.2671125, 0.49105], 0.5428571428571429: [0.3504625], 0.6571428571428573: [0.421825], 0.7714285714285716: [0.4627375], 0.8857142857142857: [0.4893], 0.999: [0.4966875]}
        # metric['BKLC'] =  {0.001: [0.0], 0.0408: [0.0004, 0.006425], 0.0806: [0.002675, 0.0384625], 0.12040000000000001: [0.00795, 0.093675], 0.1602: [0.0177625, 0.158], 0.2: [0.0304875, 0.220175], 0.3142857142857143: [0.0955875, 0.3583875], 0.4285714285714286: [0.1831625, 0.4505375], 0.5428571428571429: [0.273175], 0.6571428571428573: [0.3501875], 0.7714285714285716: [0.4146], 0.8857142857142857: [0.4603375], 0.999: [0.5000875]}
        # metric['auto-NN-BAC-non-inter'] = {0.001: [0.00025833333333333334], 0.029428571428571432: [0.0012333333333333332, 0.020183333333333334], 0.057857142857142864: [0.005325, 0.12000833333333333], 0.0862857142857143: [0.009654166666666667, 0.17714583333333334], 0.11471428571428573: [0.018058333333333332, 0.23439583333333333], 0.14314285714285716: [0.0283875, 0.28815], 0.1715714285714286: [0.04279583333333333, 0.3326375], 0.2: [0.060583333333333336, 0.3740041666666667], 0.3142857142857143: [0.1538875, 0.4639875], 0.4285714285714286: [0.2642791666666667, 0.496025], 0.5428571428571429: [0.36888333333333334], 0.6571428571428573: [0.4398208333333333], 0.7714285714285716: [0.478], 0.8857142857142857: [0.4950125], 0.999: [0.5007458333333333]}
        # metric['auto-NN-BSC-non-inter_array'] = {0.001: [0.001375], 0.01: [0.0040625, 0.00665], 0.019000000000000003: [0.0065875, 0.0122625], 0.028000000000000004: [0.009025, 0.0175375], 0.037000000000000005: [0.0127375, 0.0263125], 0.046000000000000006: [0.0155625, 0.0337], 0.05500000000000001: [0.01935, 0.041225], 0.064: [0.0228625, 0.051075], 0.073: [0.0261625, 0.0631625], 0.082: [0.0309875, 0.0705375], 0.09100000000000001: [0.034375, 0.080325], 0.1: [0.0399125, 0.09505], 0.1642857142857143:[0.0739875, 0.1726875], 0.2285714285714286: [0.114125, 0.2512], 0.2928571428571429: [0.14975, 0.32115], 0.3571428571428572: [0.19105, 0.38555], 0.4214285714285715: [0.2303, 0.4398625], 0.48571428571428577: [0.269275, 0.4872625], 0.55:[0.3012375], 0.6142857142857143: [0.340425], 0.6785714285714286: [0.3697875], 0.7428571428571429: [0.4048875], 0.8071428571428572: [0.425275], 0.8714285714285716: [0.4504], 0.9357142857142858: [0.4731875], 0.999: [0.4983625]}
        # metric['auto-NN-BSC-array-inter'] =  {0.001: [0.00030125], 0.01: [0.00176, 0.00353125], 0.019000000000000003: [0.00357625, 0.0075075], 0.028000000000000004: [0.00534125, 0.01298625], 0.037000000000000005: [0.00765875, 0.0194175], 0.046000000000000006: [0.00997, 0.02618375], 0.05500000000000001: [0.01239375, 0.03380875], 0.064: [0.01559125, 0.04246875], 0.073: [0.01832375, 0.05255625], 0.082: [0.02163125, 0.06222875], 0.09100000000000001: [0.02517125, 0.0733325], 0.1: [0.02869375, 0.08382375], 0.1642857142857143: [0.0596775, 0.16836625], 0.2285714285714286: [0.0981575, 0.25481125], 0.2928571428571429: [0.1406525, 0.3275575], 0.3571428571428572: [0.18377375, 0.3906425], 0.4214285714285715: [0.22523, 0.443985], 0.48571428571428577: [0.26659, 0.48994125], 0.55: [0.30335375], 0.6142857142857143: [0.33694875], 0.6785714285714286: [0.36916625], 0.7428571428571429: [0.39826375], 0.8071428571428572: [0.424055], 0.8714285714285716: [0.44994], 0.9357142857142858: [0.4758025], 0.999: [0.50098875]}
        # metric['auto-NN-BSC-array-one-inter-alt'] =  {0.001: [2.875e-05], 0.01: [0.0003275, 0.0008875], 0.019000000000000003: [0.00080125, 0.0029625], 0.028000000000000004: [0.00185125, 0.00677625], 0.037000000000000005: [0.00260375, 0.011605], 0.046000000000000006: [0.00381125, 0.0176125], 0.05500000000000001: [0.00566, 0.02523375], 0.064: [0.00765125, 0.03451625], 0.073: [0.00979125, 0.04380875], 0.082: [0.012545, 0.054765], 0.09100000000000001: [0.01542375, 0.0654575], 0.1: [0.01890625, 0.079795], 0.1642857142857143: [0.04988125, 0.18067875], 0.2285714285714286: [0.09477875, 0.27618125], 0.2928571428571429: [0.145505, 0.35621625], 0.3571428571428572: [0.1986375, 0.41384875], 0.4214285714285715: [0.2496475, 0.45762625],0.48571428571428577: [0.2975225, 0.49295375], 0.55: [0.33974375], 0.6142857142857143: [0.37526625], 0.6785714285714286: [0.4083075], 0.7428571428571429: [0.433925], 0.8071428571428572: [0.456805], 0.8714285714285716: [0.47470375], 0.9357142857142858: [0.48953625], 0.999: [0.499805]}
        # metric['auto-NN-BAC-array-one-inter-alt'] =  {0.001: [0.00015], 0.01: [0.000295, 0.00226625], 0.019000000000000003: [0.00056125, 0.0063525], 0.028000000000000004: [0.00101375, 0.0111875], 0.037000000000000005: [0.002105, 0.03445], 0.046000000000000006:[0.002605, 0.045195], 0.05500000000000001: [0.00372375, 0.05750125], 0.064: [0.00454625, 0.06938125], 0.073: [0.00610625, 0.08241], 0.082: [0.00727875, 0.09618875], 0.09100000000000001: [0.00944625, 0.10944375], 0.1: [0.0107275, 0.1226975], 0.1642857142857143: [0.03135, 0.21975], 0.2285714285714286: [0.06265375, 0.304125], 0.2928571428571429: [0.10410125, 0.37059125], 0.3571428571428572: [0.15119, 0.42109625], 0.4214285714285715: [0.20294125, 0.45932125], 0.48571428571428577: [0.25388375, 0.49361125], 0.55: [0.3012325], 0.6142857142857143: [0.34090125], 0.6785714285714286: [0.37993875], 0.7428571428571429: [0.40896625], 0.8071428571428572: [0.43733875], 0.8714285714285716: [0.46005625], 0.9357142857142858: [0.48029125], 0.999: [0.49912875]}
        # metric['auto-NN-BAC-array-array-inter-alt'] =  {0.001: [0.00097125], 0.020900000000000002: [0.00120125, 0.0208375], 0.0408: [0.001865, 0.041045], 0.060700000000000004: [0.002835, 0.06075125], 0.0806: [0.00418125, 0.08060625], 0.1005: [0.00602, 0.10047375], 0.12040000000000001: [0.00803125, 0.1201975], 0.1403: [0.0108325, 0.1403475], 0.1602: [0.01385875, 0.1602225], 0.1801: [0.017175, 0.1804075], 0.2: [0.02082875, 0.19997], 0.3142857142857143: [0.050085, 0.31400375], 0.4285714285714286: [0.09245875, 0.4299125], 0.5428571428571429: [0.1483625], 0.6571428571428573: [0.216815], 0.7714285714285716: [0.29734375], 0.8857142857142857: [0.3919375], 0.999: [0.4994575]}
        # metric['auto-NN-BAC-array-array-alt'] =  {0.001: [0.0010225], 0.01: [0.00101375, 0.0099125], 0.019000000000000003: [0.00119375, 0.01908375], 0.028000000000000004: [0.0013925, 0.02816625], 0.037000000000000005: [0.0016325, 0.037095], 0.046000000000000006: [0.00211875, 0.04594625], 0.05500000000000001: [0.00255875, 0.05508375], 0.064: [0.00300875, 0.0642325], 0.073: [0.00361, 0.07338625], 0.082: [0.00431125, 0.082215], 0.09100000000000001: [0.0051, 0.0908025], 0.1: [0.00587875, 0.10018125],0.1642857142857143: [0.0146375, 0.164005], 0.2285714285714286: [0.02666625, 0.22880625], 0.2928571428571429: [0.0441125, 0.29304625], 0.3571428571428572: [0.06498625, 0.35616375], 0.4214285714285715: [0.09039, 0.4216975], 0.48571428571428577: [0.11911125, 0.4850075], 0.55: [0.15262375], 0.6142857142857143: [0.18912], 0.6785714285714286: [0.23153625], 0.7428571428571429: [0.27651375], 0.8071428571428572: [0.324975], 0.8714285714285716: [0.38155375], 0.9357142857142858: [0.43890125], 0.999: [0.49946375]}

    elif N == 8:
      if k == 4:
        a=k
        # metric['BKLC-NN'] =  {0.001: [0.0], 0.0016378937069540646: [0.0, 0.0], 0.0026826957952797246: [0.0, 0.000375], 0.004393970560760791: [0.00025, 0.000625], 0.0071968567300115215: [0.0, 0.000875], 0.011787686347935873: [0.0, 0.00075], 0.019306977288832496: [0.0015, 0.003875], 0.03162277660168379: [0.000375, 0.008], 0.0517947467923121: [0.003, 0.02175], 0.08483428982440717: [0.014375, 0.052375], 0.13894954943731375: [0.02925, 0.133875], 0.22758459260747887: [0.08025, 0.254125], 0.3727593720314938: [0.178375, 0.412875], 0.6105402296585326: [0.349], 0.999: [0.512125]}
        # metric['auto-NN-std'] = {0.001: [0.0022], 0.0016378937069540646: [0.0023, 0.00345], 0.0026826957952797246: [0.0023, 0.0039], 0.004393970560760791: [0.003, 0.00585], 0.0071968567300115215: [0.00485, 0.0085], 0.011787686347935873: [0.0066, 0.01525], 0.019306977288832496: [0.009, 0.02705], 0.03162277660168379: [0.01265, 0.04075], 0.0517947467923121: [0.01785, 0.0644], 0.08483428982440717: [0.0336, 0.11355], 0.13894954943731375: [0.05625, 0.17795], 0.22758459260747887: [0.0952, 0.2947], 0.3727593720314938: [0.1816, 0.4263], 0.6105402296585326: [0.33015], 0.999: [0.49985]}
        # metric['auto-NN-selu'] = {0.001: [0.0021666666666666666], 0.0016378937069540646: [0.001, 0.0015833333333333333], 0.0026826957952797246: [0.0021666666666666666, 0.00275], 0.004393970560760791: [0.0008333333333333334, 0.006833333333333334],0.0071968567300115215: [0.0020833333333333333, 0.011166666666666667], 0.011787686347935873: [0.0025, 0.013333333333333334], 0.019306977288832496: [0.00275, 0.028166666666666666], 0.03162277660168379: [0.00425, 0.03775], 0.0517947467923121: [0.0065, 0.06308333333333334], 0.08483428982440717: [0.010416666666666666, 0.10258333333333333], 0.13894954943731375: [0.026833333333333334, 0.17916666666666667], 0.22758459260747887: [0.06391666666666666, 0.2995], 0.3727593720314938: [0.15058333333333335, 0.4200833333333333], 0.6105402296585326: [0.3179166666666667], 0.999: [0.4964166666666667]}
        # metric['auto-NN-softplus'] = {0.001: [0.0010833333333333333], 0.0408: [0.005666666666666667, 0.051083333333333335], 0.0806: [0.013416666666666667, 0.10283333333333333], 0.12040000000000001: [0.022916666666666665, 0.18816666666666668], 0.1602:[0.033916666666666664, 0.23666666666666666], 0.2: [0.0515, 0.28075], 0.3142857142857143: [0.11166666666666666, 0.4095], 0.4285714285714286: [0.17058333333333334, 0.47541666666666665], 0.5428571428571429: [0.25475], 0.6571428571428573:[0.32525], 0.7714285714285716: [0.38958333333333334], 0.8857142857142857: [0.4469166666666667], 0.999: [0.5020833333333333]}
        # metric['auto-NN-softplus_rounding'] = {0.001: [0.00015], 0.0016378937069540646: [0.0004, 0.00065], 0.0026826957952797246: [0.0005, 0.00065], 0.004393970560760791: [0.00095, 0.0014], 0.0071968567300115215: [0.0015, 0.0031], 0.011787686347935873: [0.00165, 0.00385], 0.019306977288832496: [0.00375, 0.0078], 0.03162277660168379: [0.00475, 0.01495], 0.0517947467923121: [0.01135, 0.0369], 0.08483428982440717: [0.0226, 0.06905], 0.13894954943731375: [0.05035, 0.1494], 0.22758459260747887: [0.1018, 0.27835], 0.3727593720314938: [0.2012, 0.4257], 0.6105402296585326: [0.35305], 0.999: [0.4942]}
        # metric['auto-NN-lk_relu_0.5'] = {0.001: [0.0013], 0.0016378937069540646: [0.0009, 0.0024], 0.0026826957952797246: [0.0013, 0.00445], 0.004393970560760791: [0.00195, 0.0053], 0.0071968567300115215: [0.0011, 0.01005], 0.011787686347935873: [0.00105, 0.01565], 0.019306977288832496: [0.0022, 0.0253], 0.03162277660168379: [0.00225, 0.04265], 0.0517947467923121: [0.00435, 0.07105], 0.08483428982440717: [0.0085, 0.1145], 0.13894954943731375: [0.02075, 0.2008], 0.22758459260747887: [0.05575, 0.30915], 0.3727593720314938: [0.13625, 0.44395], 0.6105402296585326: [0.3085], 0.999: [0.481]}
        # metric['auto-NN-mish'] = {0.001: [0.0], 0.0016378937069540646: [0.0005833333333333334, 0.00125], 0.0026826957952797246: [0.00075, 0.002], 0.004393970560760791: [0.0006666666666666666, 0.004833333333333334], 0.0071968567300115215: [0.0013333333333333333, 0.006833333333333334], 0.011787686347935873: [0.0006666666666666666, 0.0095], 0.019306977288832496: [0.0010833333333333333, 0.015666666666666666], 0.03162277660168379: [0.003416666666666667, 0.028], 0.0517947467923121: [0.005166666666666667, 0.05291666666666667], 0.08483428982440717: [0.010416666666666666, 0.09833333333333333], 0.13894954943731375: [0.028166666666666666, 0.17566666666666667], 0.22758459260747887: [0.0665, 0.2915], 0.3727593720314938:[0.14833333333333334, 0.42783333333333334], 0.6105402296585326: [0.30425], 0.999: [0.5180833333333333]}
        # metric['auto-NN-softplus-norm_training'] = {0.001: [0.0022], 0.0408: [0.00315, 0.0615], 0.0806: [0.0076, 0.1191], 0.12040000000000001: [0.01615, 0.1769], 0.1602: [0.0268, 0.23625], 0.2: [0.0447, 0.2847], 0.3142857142857143: [0.10295, 0.39915], 0.4285714285714286: [0.1754, 0.4801], 0.5428571428571429: [0.2539], 0.6571428571428573: [0.3457], 0.7714285714285716: [0.42535], 0.8857142857142857: [0.46995], 0.999: [0.51255]}
        # metric['auto-NN-BSC-inter'] = {0.001: [7.5e-05], 0.020900000000000002: [0.0043, 0.0085], 0.0408: [0.009625, 0.023075], 0.060700000000000004: [0.0119, 0.03995], 0.0806: [0.020525, 0.06355], 0.1005: [0.030475, 0.0864], 0.12040000000000001: [0.0394,0.114025], 0.1403: [0.045225, 0.1441], 0.1602: [0.05885, 0.1729], 0.1801: [0.076, 0.203325], 0.2: [0.08185, 0.229025], 0.3142857142857143: [0.167275, 0.37595], 0.4285714285714286: [0.24185, 0.4667], 0.5428571428571429: [0.314725], 0.6571428571428573: [0.383675], 0.7714285714285716: [0.43425], 0.8857142857142857: [0.472675], 0.999: [0.51705]}
        # 'auto-NN-BSC' model for the BSC, intervals definition train_epsilon = 0.05
        # metric['auto-NN-BAC-not-inter'] = {0.001: [0.00105], 0.0408: [0.0068, 0.0555], 0.0806: [0.0146, 0.119], 0.12040000000000001: [0.02415, 0.1662], 0.1602: [0.043, 0.21565], 0.2: [0.05485, 0.27165], 0.3142857142857143: [0.12145, 0.3833], 0.4285714285714286: [0.19565, 0.46235], 0.5428571428571429: [0.27575], 0.6571428571428573: [0.3596], 0.7714285714285716: [0.4206], 0.8857142857142857: [0.46855], 0.999: [0.49455]}
        # 'auto-NN-BAC' model for the BAC, non intervals definition, train_epsilon = 0.02 and epsilon_0_max = 0.2
        # metric['auto-NN-BAC-non-inter'] = {0.001: [0.00042], 0.020900000000000002: [0.00218, 0.010535], 0.0408: [0.00566, 0.026205], 0.060700000000000004: [0.010655, 0.045925], 0.0806: [0.016315, 0.069735], 0.1005: [0.022375, 0.095615], 0.12040000000000001: [0.030425, 0.124775], 0.1403: [0.038995, 0.15203], 0.1602: [0.04765, 0.17778], 0.1801: [0.05783, 0.21027], 0.2: [0.06916, 0.23848], 0.3142857142857143: [0.14124, 0.379495], 0.4285714285714286: [0.21624, 0.46712], 0.5428571428571429: [0.29385], 0.6571428571428573: [0.36633], 0.7714285714285716: [0.424145], 0.8857142857142857: [0.475765], 0.999: [0.51538]}
        # metric['auto-NN-BAC-non-inter'] model for BAC non intervals definition, train epsilon = 0.02 an epsilon_0_max = 0.07
        # metric['auto-NN-BAC-array-array-inter'] =  {0.001: [0.000865], 0.01: [0.0013425, 0.00892], 0.019000000000000003: [0.001825, 0.0169275], 0.028000000000000004: [0.0023575, 0.0252175], 0.037000000000000005: [0.0031775, 0.0339175], 0.046000000000000006: [0.004165,0.0435275], 0.05500000000000001: [0.00523, 0.053155], 0.064: [0.0061025, 0.0601775], 0.073: [0.0073975, 0.069735], 0.082: [0.0087075, 0.0786675], 0.09100000000000001: [0.01013, 0.08835], 0.1: [0.011725, 0.0967925], 0.1642857142857143:[0.0253175, 0.163865], 0.2285714285714286: [0.0445125, 0.23079], 0.2928571428571429: [0.066845, 0.297765], 0.3571428571428572: [0.0924925, 0.3596325], 0.4214285714285715: [0.121985, 0.4205575], 0.48571428571428577: [0.154935, 0.479755], 0.55: [0.1898175], 0.6142857142857143: [0.22739], 0.6785714285714286: [0.2675525], 0.7428571428571429: [0.3098725], 0.8071428571428572: [0.356425], 0.8714285714285716: [0.408065], 0.9357142857142858: [0.459835], 0.999: [0.5158075]}
        metric['auto-NN-BAC-array-array'] =  {0.001: [0.001375], 0.01: [0.0023375, 0.0140375], 0.019000000000000003: [0.003125, 0.0262875], 0.028000000000000004: [0.0042125, 0.0420125], 0.037000000000000005: [0.005975, 0.05155], 0.046000000000000006: [0.0072625,0.065825], 0.05500000000000001: [0.0087875, 0.0765875], 0.064: [0.010525, 0.0907125], 0.073: [0.012725, 0.1013], 0.082: [0.0136125, 0.111375], 0.09100000000000001: [0.0160625, 0.12655], 0.1: [0.017975, 0.13625], 0.1642857142857143: [0.03355, 0.2121375], 0.2285714285714286: [0.0542875, 0.277025], 0.2928571428571429: [0.080625, 0.3380625], 0.3571428571428572: [0.1092, 0.3935125], 0.4214285714285715: [0.1419, 0.4394125], 0.48571428571428577: [0.175925, 0.4859375], 0.55: [0.2136], 0.6142857142857143: [0.2545], 0.6785714285714286: [0.293525], 0.7428571428571429: [0.3371375], 0.8071428571428572: [0.3812], 0.8714285714285716: [0.4261125], 0.9357142857142858: [0.469325], 0.999: [0.5186]}
    elif N == 4:
      if k == 2:
        a=k
        # metric['Polar-NN'] =  {0.001: [0.00025], 0.01: [0.00135], 0.019000000000000003: [0.00165], 0.028000000000000004: [0.00165], 0.037000000000000005: [0.00125], 0.046000000000000006: [0.00275], 0.05500000000000001: [0.0035], 0.064: [0.00545], 0.073: [0.0067], 0.082: [0.0072], 0.09100000000000001: [0.00895], 0.1: [0.00875], 0.1642857142857143: [0.0228], 0.2285714285714286: [0.0425], 0.2928571428571429: [0.0718], 0.3571428571428572: [0.1051], 0.4214285714285715: [0.13295], 0.48571428571428577:[0.18575], 0.55: [0.2282], 0.6142857142857143: [0.2693], 0.6785714285714286: [0.3186], 0.7428571428571429: [0.3592], 0.8071428571428572: [0.3998], 0.8714285714285716: [0.45155], 0.9357142857142858: [0.48125], 0.999: [0.33505]}
        metric['Polar'] = {0.001: [0.00093], 0.01: [0.001345], 0.019000000000000003: [0.00139], 0.028000000000000004: [0.002065], 0.037000000000000005: [0.00219], 0.046000000000000006: [0.00322], 0.05500000000000001: [0.00416], 0.064: [0.00481], 0.073: [0.005695], 0.082: [0.006755], 0.09100000000000001: [0.008475], 0.1: [0.00923], 0.1642857142857143: [0.02412], 0.2285714285714286: [0.044765], 0.2928571428571429: [0.07188], 0.3571428571428572: [0.102675], 0.4214285714285715: [0.137715], 0.48571428571428577: [0.179545], 0.55: [0.223735], 0.6142857142857143: [0.26784], 0.6785714285714286: [0.31448], 0.7428571428571429: [0.358775], 0.8071428571428572: [0.40254], 0.8714285714285716: [0.441855], 0.9357142857142858: [0.47655], 0.999: [0.333565]}
    return metric
  elif graph == 'BLER':
    if N == 16:
      if k == 4:
        a=k
        # metric['auton-NN-BSC-non-inter_array'] = {0.001: [0.0], 0.01: [8e-05, 0.00016], 0.019000000000000003: [0.00015, 0.00087], 0.028000000000000004: [0.00031, 0.00173], 0.037000000000000005: [0.0005, 0.00316], 0.046000000000000006: [0.001, 0.00543], 0.05500000000000001: [0.00167, 0.00842], 0.064: [0.00225, 0.01144], 0.073: [0.00303, 0.01697], 0.082: [0.00425, 0.02336], 0.09100000000000001: [0.00519, 0.03005], 0.1: [0.0065, 0.03888], 0.1642857142857143: [0.02642, 0.13889], 0.2285714285714286:[0.06371, 0.30559], 0.2928571428571429: [0.11949, 0.50425], 0.3571428571428572: [0.19604, 0.68944], 0.4214285714285715: [0.28341, 0.83179], 0.48571428571428577: [0.37599, 0.92281], 0.55: [0.46912], 0.6142857142857143: [0.55888], 0.6785714285714286: [0.63618], 0.7428571428571429: [0.71178], 0.8071428571428572: [0.77645], 0.8714285714285716: [0.83306], 0.9357142857142858: [0.88593], 0.999: [0.93253]}
        # metric['auton-NN-BAC-non-inter_array'] =  {0.001: [0.0], 0.020900000000000002: [2e-05, 0.00092], 0.0408: [0.00038, 0.00446], 0.060700000000000004: [0.00053, 0.01264], 0.0806: [0.00251, 0.0351], 0.1005: [0.00478, 0.06004], 0.12040000000000001: [0.00705, 0.09461], 0.1403: [0.01101, 0.13265], 0.1602: [0.01624, 0.17883], 0.1801: [0.02141, 0.23045], 0.2: [0.02956, 0.28749], 0.3142857142857143: [0.10337, 0.61764], 0.4285714285714286: [0.2349, 0.85685], 0.5428571428571429: [0.4045], 0.6571428571428573: [0.58602], 0.7714285714285716: [0.7439], 0.8857142857142857: [0.85691], 0.999: [0.93273]}

      elif k == 8:
        # metric['Uncoded']= {0.001: [0.0018375], 0.001995262314968879: [0.00208125, 0.00203125], 0.003981071705534973: [0.0029375, 0.00398125], 0.007943282347242814: [0.0049375, 0.00785], 0.015848931924611134: [0.0091, 0.015825], 0.03162277660168379: [0.0171375, 0.032675], 0.0630957344480193: [0.03314375, 0.0641625], 0.12589254117941676: [0.06451875, 0.12606875], 0.25118864315095796: [0.12760625, 0.25244375], 0.501187233627272: [0.251975], 0.999: [0.50184375]}
        # metric['Polar']= {0.001: [5.625e-05], 0.001995262314968879: [0.00013125, 0.000125], 0.003981071705534973: [0.000175, 0.00043125], 0.007943282347242814: [0.00063125, 0.0016625], 0.015848931924611134: [0.00165, 0.005325], 0.03162277660168379: [0.00366875, 0.021475], 0.0630957344480193: [0.0136625, 0.0685], 0.12589254117941676: [0.04769375, 0.1856], 0.25118864315095796: [0.1312375, 0.37719375], 0.501187233627272: [0.32815], 0.999: [0.49995625]}
        # metric['BCH']= {0.001: [8.75e-05], 0.001995262314968879: [6.25e-05, 5e-05], 0.003981071705534973: [0.00013125, 0.00026875], 0.007943282347242814: [0.00023125, 0.00080625], 0.015848931924611134: [0.00044375, 0.00365], 0.03162277660168379: [0.0018125, 0.01293125], 0.0630957344480193: [0.007025, 0.0442875], 0.12589254117941676: [0.0258375, 0.1336625], 0.25118864315095796: [0.09119375, 0.2986625], 0.501187233627272: [0.264575], 0.999: [0.50145]}
        # metric['BKLC']= {0.001: [1.875e-05], 0.001995262314968879: [0.0, 0.0], 0.003981071705534973: [0.0, 0.0], 0.007943282347242814: [2.5e-05, 0.00011875], 0.015848931924611134: [8.125e-05, 0.00055], 0.03162277660168379: [0.00045625, 0.00354375], 0.0630957344480193: [0.001875, 0.02281875], 0.12589254117941676: [0.010375, 0.10095], 0.25118864315095796: [0.05811875, 0.2904875], 0.501187233627272: [0.241625], 0.999: [0.49931875]}
        # metric['L+M']= {0.001: [0.00028125], 0.001995262314968879: [0.0002875, 0.0004875], 0.003981071705534973: [0.00031875, 0.00074375], 0.007943282347242814: [0.00074375, 0.00180625], 0.015848931924611134: [0.0014875, 0.0053125], 0.03162277660168379: [0.00279375, 0.01525625], 0.0630957344480193: [0.00936875, 0.0501125], 0.12589254117941676: [0.0276875, 0.16263125], 0.25118864315095796: [0.09575, 0.36948125], 0.501187233627272: [0.30030625], 0.999: [0.50085]}
        # metric['P+M']= {0.001: [0.02591875], 0.001995262314968879: [0.026875, 0.0267875], 0.003981071705534973: [0.02643125, 0.02705625], 0.007943282347242814: [0.029975, 0.03229375], 0.015848931924611134: [0.0315, 0.040325], 0.03162277660168379: [0.0385125, 0.05834375], 0.0630957344480193: [0.0501875, 0.10281875], 0.12589254117941676: [0.0769625, 0.20686875], 0.25118864315095796: [0.15371875, 0.39544375], 0.501187233627272: [0.32949375], 0.999: [0.49814375]}
        # metric['Int']= {0.001: [0.12483125], 0.001995262314968879: [0.1248375, 0.1267125], 0.003981071705534973: [0.124425, 0.1256], 0.007943282347242814: [0.1253, 0.126575], 0.015848931924611134: [0.1259125, 0.1279125], 0.03162277660168379: [0.1276875, 0.13408125], 0.0630957344480193: [0.1288375, 0.155725], 0.12589254117941676: [0.14586875, 0.2274], 0.25118864315095796: [0.19374375, 0.3874875], 0.501187233627272: [0.337825], 0.999: [0.50081875]}
        # metric['auto-NN-BAC-non-inter'] =  {0.001: [0.00046666666666666666], 0.029428571428571432: [0.0024666666666666665, 0.03933333333333333], 0.057857142857142864: [0.011033333333333332, 0.24146666666666666], 0.0862857142857143: [0.0194, 0.3554333333333333], 0.11471428571428573: [0.0367, 0.47046666666666664], 0.14314285714285716: [0.0572, 0.5758333333333333], 0.1715714285714286: [0.08626666666666667, 0.6642666666666667], 0.2: [0.1215, 0.7457], 0.3142857142857143: [0.30823333333333336, 0.9269], 0.4285714285714286: [0.5296333333333333, 0.987], 0.5428571428571429: [0.7356333333333334], 0.6571428571428573: [0.8733333333333333], 0.7714285714285716: [0.9504333333333334], 0.8857142857142857: [0.9842666666666666], 0.999: [0.9957333333333334]}
        # metric['auto-NN-BSC-non-inter_array'] = {0.001: [0.0108], 0.01: [0.0299, 0.0477], 0.019000000000000003: [0.0461, 0.0841], 0.028000000000000004: [0.0645, 0.1145], 0.037000000000000005: [0.0855, 0.1629], 0.046000000000000006: [0.1054, 0.201], 0.05500000000000001: [0.1273, 0.2375], 0.064: [0.1432, 0.2821], 0.073: [0.1623, 0.3373], 0.082: [0.1903, 0.3709], 0.09100000000000001: [0.2082, 0.4086], 0.1: [0.2319, 0.4544], 0.1642857142857143: [0.3842, 0.6953], 0.2285714285714286: [0.5251, 0.834],0.2928571428571429: [0.6321, 0.9221], 0.3571428571428572: [0.731, 0.9677], 0.4214285714285715: [0.812, 0.986], 0.48571428571428577: [0.8722, 0.9942], 0.55: [0.905], 0.6142857142857143: [0.9431], 0.6785714285714286: [0.9604], 0.7428571428571429: [0.9794], 0.8071428571428572: [0.9846], 0.8714285714285716: [0.9918], 0.9357142857142858: [0.9937], 0.999: [0.9956]}
        # metric['auto-NN-BSC-array-inter'] =  {0.001: [0.00167], 0.01: [0.00975, 0.01927], 0.019000000000000003: [0.01955, 0.03831], 0.028000000000000004: [0.02793, 0.06284], 0.037000000000000005: [0.0393, 0.09026], 0.046000000000000006: [0.04927, 0.11793], 0.05500000000000001: [0.06126, 0.14867], 0.064: [0.07402, 0.18095], 0.073: [0.0863, 0.21653], 0.082: [0.09987, 0.24951], 0.09100000000000001: [0.11508, 0.28791], 0.1: [0.12834, 0.32485], 0.1642857142857143: [0.24418, 0.5674], 0.2285714285714286: [0.36614, 0.76321], 0.2928571428571429: [0.48996, 0.88472], 0.3571428571428572: [0.60519, 0.95296], 0.4214285714285715: [0.70028, 0.98246], 0.48571428571428577: [0.78466, 0.99472], 0.55: [0.84699], 0.6142857142857143: [0.89642], 0.6785714285714286: [0.93304], 0.7428571428571429: [0.95738], 0.8071428571428572: [0.97451], 0.8714285714285716: [0.98542], 0.9357142857142858: [0.99227], 0.999: [0.99591]}
        # metric['auto-NN-BSC-array-one-inter-alt'] =  {0.001: [0.00014], 0.01: [0.00181, 0.00405], 0.019000000000000003: [0.00372, 0.01189], 0.028000000000000004: [0.00759, 0.0258], 0.037000000000000005: [0.01087, 0.04186], 0.046000000000000006: [0.01521, 0.06274], 0.05500000000000001: [0.0216, 0.08646], 0.064: [0.02861, 0.11573], 0.073: [0.03604, 0.14419], 0.082: [0.04579, 0.17561], 0.09100000000000001: [0.05518, 0.20689], 0.1: [0.06574, 0.24837], 0.1642857142857143: [0.16178, 0.51121], 0.2285714285714286: [0.28778, 0.72797], 0.2928571428571429: [0.4208, 0.86995], 0.3571428571428572: [0.55329, 0.94732], 0.4214285714285715: [0.66507, 0.9811], 0.48571428571428577: [0.76018, 0.99487], 0.55: [0.83559], 0.6142857142857143: [0.89228], 0.6785714285714286: [0.93247], 0.7428571428571429: [0.95927], 0.8071428571428572: [0.97623], 0.8714285714285716: [0.98551], 0.9357142857142858: [0.99148], 0.999: [0.99635]}
        # metric['auto-NN-BAC-array-one-inter-alt'] =  {0.001: [0.00078], 0.01: [0.0015, 0.01107], 0.019000000000000003: [0.00247, 0.02686], 0.028000000000000004: [0.00419, 0.04509], 0.037000000000000005: [0.00801, 0.12373], 0.046000000000000006: [0.01041, 0.15744], 0.05500000000000001: [0.01378, 0.19666], 0.064: [0.01698, 0.23251], 0.073: [0.02239, 0.27143], 0.082: [0.02615, 0.31065], 0.09100000000000001: [0.0332, 0.34671], 0.1: [0.03783, 0.38242], 0.1642857142857143: [0.10256, 0.62234], 0.2285714285714286: [0.19563, 0.79679], 0.2928571428571429: [0.31108, 0.90166], 0.3571428571428572: [0.43549, 0.95794], 0.4214285714285715: [0.56234, 0.98395], 0.48571428571428577: [0.67586, 0.99456], 0.55: [0.77299], 0.6142857142857143:[0.84857], 0.6785714285714286: [0.90672], 0.7428571428571429: [0.94272], 0.8071428571428572: [0.96887], 0.8714285714285716: [0.98447], 0.9357142857142858: [0.99189], 0.999: [0.99578]}
        # metric['auto-NN-BAC-array-array-inter-alt'] =  {0.001: [0.00773], 0.020900000000000002: [0.00954, 0.1553], 0.0408: [0.01485, 0.28543], 0.060700000000000004: [0.02244, 0.39366], 0.0806: [0.03285, 0.49032], 0.1005: [0.04707, 0.57092], 0.12040000000000001: [0.06245,0.63822], 0.1403: [0.0836, 0.70134], 0.1602: [0.10593, 0.75243], 0.1801: [0.12942, 0.79774], 0.2: [0.15437, 0.83143], 0.3142857142857143: [0.33719, 0.95195], 0.4285714285714286: [0.54125, 0.98851], 0.5428571428571429: [0.72276], 0.6571428571428573: [0.85935], 0.7714285714285716: [0.94048], 0.8857142857142857: [0.98169], 0.999: [0.99606]}
        metric['auto-NN-BAC-array-array-alt'] =  {0.001: [0.00815], 0.01: [0.00809, 0.07645], 0.019000000000000003: [0.00952, 0.1427], 0.028000000000000004: [0.01106, 0.20441], 0.037000000000000005: [0.01291, 0.26033], 0.046000000000000006: [0.01681, 0.31341], 0.05500000000000001: [0.02032, 0.36548], 0.064: [0.02385, 0.41279], 0.073: [0.02852, 0.45546], 0.082: [0.0339, 0.49542], 0.09100000000000001: [0.04011, 0.53349], 0.1: [0.0462, 0.5717], 0.1642857142857143: [0.11113, 0.76133], 0.2285714285714286: [0.19396, 0.8743], 0.2928571428571429: [0.30317, 0.93738], 0.3571428571428572: [0.41583, 0.97023], 0.4214285714285715: [0.5306, 0.98752], 0.48571428571428577: [0.63878, 0.99521], 0.55: [0.73486], 0.6142857142857143: [0.81308], 0.6785714285714286: [0.87761], 0.7428571428571429: [0.92502], 0.8071428571428572: [0.95661], 0.8714285714285716: [0.97804], 0.9357142857142858: [0.9901], 0.999: [0.99606]}
    elif N == 8:
      if k == 4:
        a=k
        # metric['auto-NN-BSC-inter'] = {0.001: [0.0002], 0.020900000000000002: [0.008, 0.0167], 0.0408: [0.0181, 0.044], 0.060700000000000004: [0.0252, 0.0773], 0.0806: [0.0438, 0.1249], 0.1005: [0.0642, 0.1687], 0.12040000000000001: [0.0818, 0.2201], 0.1403: [0.0932, 0.276], 0.1602: [0.1183, 0.3309], 0.1801: [0.1537, 0.3903], 0.2: [0.1674, 0.4382], 0.3142857142857143: [0.3287, 0.7111], 0.4285714285714286: [0.47, 0.8769], 0.5428571428571429: [0.6134], 0.6571428571428573: [0.7351], 0.7714285714285716: [0.8213], 0.8857142857142857: [0.883], 0.999: [0.9362]}
        # metric['auto-NN-BAC-non-inter'] =  {0.001: [0.00066], 0.020900000000000002: [0.0043, 0.01802], 0.0408: [0.01092, 0.04682], 0.060700000000000004: [0.02086, 0.0824], 0.0806: [0.03156, 0.12604], 0.1005: [0.04336, 0.17406], 0.12040000000000001: [0.05886,0.2277], 0.1403: [0.07624, 0.28056], 0.1602: [0.0927, 0.32924], 0.1801: [0.11114, 0.38966], 0.2: [0.13418, 0.44092], 0.3142857142857143: [0.27238, 0.70612], 0.4285714285714286: [0.41934, 0.87394], 0.5428571428571429: [0.56816], 0.6571428571428573: [0.70292], 0.7714285714285716: [0.80544], 0.8857142857142857: [0.88458], 0.999: [0.9324]}
        # metric['auto-NN-BAC-array-array-inter'] =  {0.001: [0.00338], 0.01: [0.00521, 0.03428], 0.019000000000000003: [0.00694, 0.06423], 0.028000000000000004: [0.00903, 0.09366], 0.037000000000000005: [0.01198, 0.12403], 0.046000000000000006: [0.01569, 0.1576], 0.05500000000000001: [0.01901, 0.18804], 0.064: [0.02219, 0.20918], 0.073: [0.02681, 0.23871], 0.082: [0.03177, 0.26678], 0.09100000000000001: [0.03657, 0.29439], 0.1: [0.04197, 0.31763], 0.1642857142857143: [0.08825, 0.48985], 0.2285714285714286: [0.15135, 0.6273], 0.2928571428571429: [0.22232, 0.73768], 0.3571428571428572: [0.29791, 0.81905], 0.4214285714285715: [0.37974, 0.88392], 0.48571428571428577: [0.46461, 0.92596], 0.55: [0.54399], 0.6142857142857143: [0.623],0.6785714285714286: [0.69462], 0.7428571428571429: [0.75837], 0.8071428571428572: [0.81433], 0.8714285714285716: [0.86547], 0.9357142857142858: [0.90379], 0.999: [0.93224]}
        metric['auto-NN-BAC-array-array'] =  {0.001: [0.00375], 0.01: [0.00665, 0.0389], 0.019000000000000003: [0.0086, 0.0733], 0.028000000000000004: [0.01165, 0.1158], 0.037000000000000005: [0.0163, 0.1408], 0.046000000000000006: [0.02005, 0.1739], 0.05500000000000001: [0.0242, 0.2063], 0.064: [0.0292, 0.2391], 0.073: [0.0336, 0.26785], 0.082: [0.0375, 0.29035], 0.09100000000000001: [0.04335, 0.3271], 0.1: [0.04825, 0.3509], 0.1642857142857143: [0.08995, 0.51875], 0.2285714285714286: [0.14535, 0.64965], 0.2928571428571429: [0.21245, 0.7513], 0.3571428571428572: [0.2829, 0.83115], 0.4214285714285715: [0.3598, 0.88775], 0.48571428571428577: [0.4391, 0.92955], 0.55: [0.5185], 0.6142857142857143: [0.5969], 0.6785714285714286: [0.6717], 0.7428571428571429: [0.74355], 0.8071428571428572: [0.8013], 0.8714285714285716: [0.8549], 0.9357142857142858: [0.897], 0.999: [0.93125]}
    return metric

def read_ber_file(N,k,graph):
  if (N == 16 or N == 8) and (k == 4 or k==8):

    if graph == 'BLER':
      file = open(f"./Data_plots/data_({N},{k})_BLER.txt", "r")
    else:
      file = open(f"./Data_plots/data_({N},{k})_100000pkts.txt", "r")
    all_BER = eval(file.read())
    BER = {}
    # 'Flip(2,4,6)' 'BCH(0)' ,'Uncode' 'Polar(0.1)', 'Flip(2,4,6)', 'L+M(0.55)', 'P(0.1)+M(0.5)' ,'BKLC'
    for a in ['Polar(0.1)']: #List of codes you want to plot from file
      BER[a] = all_BER[a]
    file.close()
  else:
    BER = {}
  return BER

def write_ber_file(N,k,metric,nb_pkts):
  file = open(f"data_({N},{k})_{nb_pkts}pkts.txt", "w")
  print(str(metric))
  file.write(str(metric))
  file.close()

def polar_generator_matrix(N=8, k=4, channel_type='AWGN', design_parameter=0.1):
  # initialise polar code
  myPC = PolarCode(N, k)
  myPC.construction_type = 'bb'

  Construct(myPC, design_parameter)

  T = arikan_gen(int(np.log2(N)))
  infoBits = [myPC.reliabilities[a] for a in range(len(myPC.reliabilities) - 1, len(myPC.reliabilities) - k - 1, -1)]
  infoBits.sort()
  G = []

  for j in range(len(T)):
    G.append([T[j][i] for i in infoBits])

  G = np.array(G)

  return np.transpose(G), infoBits
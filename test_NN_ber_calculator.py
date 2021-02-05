#!/usr/bin/python
import sys
import numpy as np
import polar_codes_generator as polar
import utils
import utils_ML
import matrix_codes as mat_gen
import keras
import matplotlib.pyplot as plt

model_encoder = keras.models.load_model("autoencoder/model_encoder.h5", compile=False)
model_decoder = keras.models.load_model("autoencoder/model_decoder.h5", compile=False)
k = 8
N = 16
MAP_test = False
u_k = utils.symbols_generator(k)
x_n = model_encoder.predict(np.array(u_k))
u_hat_k = np.array(model_decoder(x_n)).tolist()
# for i in range(2**k):
#   print(u_k[i],np.round(x_n[i]),np.round(u_hat_k[i]))

nb_pkts = 10000
e0 = np.concatenate((np.linspace(0.001, 0.2, 10, endpoint=False), np.linspace(0.2, 1, 8)), axis=0)
utils_ML.BER_NN(nb_pkts, k, N, model_encoder, model_decoder, e0, MAP_test, 'array', 'array')

plt.show()

import sys
import utils
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from keras.models import Sequential, model_from_json
import keras.backend as K
import tensorflow as tf
import keras
import time


shape = 1.5
scale = 0.1

fig = plt.figure(figsize=(7, 3.5), dpi=180, facecolor='w', edgecolor='k')
fig.subplots_adjust(wspace=0.4, top=0.8)
fig.suptitle('Distributions', fontsize=14)
ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

batch_size = 10000
epsilon0_train_max = 0.05
# epsilon = np.random.gamma(1, 2, size=(batch_size, 1))*0.025
# epsilon = np.random.chisquare(3, size=(batch_size, 1)) * 0.01
# epsilon = np.random.exponential(0.05, size=(batch_size, 1))
# epsilon = np.random.lognormal(mean=-3.3, sigma=1.8, size=(batch_size, 1))
epsilon = np.random.uniform(low=0.0, high=epsilon0_train_max, size=(batch_size, 1))

epsilon = [s if s< epsilon0_train_max else np.random.uniform(high=epsilon0_train_max) for s in epsilon]

ax1.hist(epsilon, 100, range=(0,0.3))

interval = [int(9.30*s**0.5) if int(9.30*s**0.5)<4 else 3 for s in epsilon]
ax2.hist(interval, 4, range=(0,3))

e0 = np.linspace(0,0.3,1001)
i = [int(9.30*s**0.5) if int(9.30*s**0.5)<4 else 3 for s in e0]
plt.figure(0)
plt.plot(e0,i)
plt.show()
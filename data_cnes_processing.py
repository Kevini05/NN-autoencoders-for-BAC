import numpy as np

import matplotlib.pyplot as plt

plt.ion()
data = np.genfromtxt("./Data CNES/noise_distributions_ook_25g.dat",dtype=None,delimiter='\t')

def getList(dict):
  list = []
  for key in dict.keys():
    list.append(key)
  return list

def plot_distribution(a):
  plt.plot(a['bin_edges'], a['prob_zero'], label='zero')
  plt.plot(a['bin_edges'], a['prob_one'], label='one')

  plt.title(f"Signal distribution for {str(rop)} dB")
  plt.ylabel('Probability')
  plt.xlabel('Power')
  plt.legend(loc="upper right")
  plt.grid()

dict_data = {}
for i in range(len(data)):
  row = data[i]
  row = row[~np.isnan(row)]

  if i%4 == 0:
    dict_row = {}
    rop = row[0]
  if i % 4 == 1:
    dict_row['bin_edges'] = row[:-1]
  if i % 4 == 2:
    dict_row['prob_zero'] = row
  if i % 4 == 3:
    dict_row['prob_one'] = row

    dict_data[str(rop)] = dict_row



keys = getList(dict_data)

list_threshold = []
list_e0 = []
list_e1 = []
list_rop = []
dict_error = {}
print("ROP \t \t e0 \t \t e1")
for i in range(len(keys)):
  list_rop.append(float(keys[i]))
  a = dict_data[keys[i]]
  error_list = []
  for t in range(len(a['prob_zero'])):
    e_threshold = sum(a['prob_zero'][t:])+sum(a['prob_one'][:t])
    error_list.append(e_threshold)
  dict_error[keys[i]] = error_list

  threshold = error_list.index(min(error_list))
  print(keys[i], a['bin_edges'][threshold])
  list_threshold.append(threshold)
  epsilon_0 = sum(a['prob_zero'][threshold:])
  epsilon_1 = sum(a['prob_one'][:threshold])
  list_e0.append(epsilon_0)
  list_e1.append(epsilon_1)

  # print(f"{'{:.5f}'.format(float(keys[i]))}\t {'{:.5f}'.format(epsilon_0)}\t {'{:.5f}'.format(epsilon_1)}")

plt.plot(list_rop,list_e0,label='$\epsilon_0$',linewidth=0.5)
plt.plot(list_rop,list_e1,label='$\epsilon_1$',linewidth=0.5)
plt.plot(list_rop,[list_e0[i]+list_e1[i] for i in range(len(list_e1))],label='$\epsilon_0$+$\epsilon_1$',linewidth=0.5)
plt.title('Crossover probabilities w.r.t. ROP')
plt.ylabel('Crossover Probability')
plt.xlabel('ROP')
plt.legend(loc="best")
plt.grid()

nb_figures = 8 #must be even
idx = np.linspace(0,len(keys)-1,nb_figures).astype(int)
title = 'Data CNES'
fig = plt.figure(figsize=(7, 3.5), dpi=180, facecolor='w', edgecolor='k')
fig.subplots_adjust(wspace=0.4, top=0.8)
fig.suptitle(title, fontsize=14)
count = 0
nb_row = 2
nb_column = int(nb_figures/nb_row)
for i in idx:
# for i in range(nb_figures):
  x_position = int(count%nb_column)
  y_position = int(count/nb_column)
  ax1 = plt.subplot2grid((nb_row, nb_column), (y_position, x_position), rowspan=1, colspan=1)
  a = dict_data[keys[i]]

  ax1.plot(a['bin_edges'], a['prob_zero'], label='P(x=0)',linewidth=0.5)
  ax1.plot(a['bin_edges'], a['prob_one'], label='P(x=1)',linewidth=0.5)
  ax1.plot(a['bin_edges'], np.array(dict_error[keys[i]])*max(a['prob_zero']), label='error', linewidth=0.5)
  ax1.vlines(a['bin_edges'][list_threshold[i]], 0, max(a['prob_zero']), linestyles="dotted", colors="r",  linewidth=0.5)
  ax1.legend(prop={'size': 5},loc="upper right")
  t = "{:.1f}".format(float(keys[i]))
  ax1.set_title(f"ROP = {t} dBm", fontsize=8)
  ax1.set_xlabel('Electric Signal', fontsize=8)
  ax1.set_ylabel('Probability', fontsize=8)
  ax1.grid(which='both', linewidth=0.2)
  count+= 1

plt.tight_layout()


fig.savefig("./figures/data_cnes_processing")

# plt.show()
# \Python3\python.exe data_cnes_processing.py

import sys
import os

program = '\Python3\python.exe'
channel = sys.argv[1]
N = int(sys.argv[2])
k = int(sys.argv[3])
iterations = int(sys.argv[4])
nb_pkts = int(sys.argv[5])
length_training = sys.argv[6]


# os.system("{program}data_cnes_processing.py")

# os.system(f"{program} autoencoder_array-array_alt-training_loss-validator_regularizer.py {N} {k} {iterations} {nb_pkts} {length_training}")
# os.system(f"{program} autoencoder_array-array_alt-training_loss-validator_regularizer_interval.py {N} {k} {iterations} {nb_pkts} {length_training}")
os.system(f"{program} autoencoder_array-array_fine-decoder.py {channel} {N} {k} {iterations} {nb_pkts} {length_training}")

os.system(f"{program} autoencoder_array-array_polar_alt-training.py {channel} {N} {k} {iterations*2} {nb_pkts} {length_training}")

os.system(f"{program} autoencoder_array-onehot_fine-decoder.py {N} {k} {nb_pkts} {length_training}")
os.system(f"{program} autoencoder_array-onehot_fine-decoder_ext-interval.py {N} {k} {nb_pkts} {length_training}")
os.system(f"{program} autoencoder_array-onehot_fine-decoder_int-interval.py {N} {k} {nb_pkts} {length_training}")
os.system(f"{program} autoencoder_array-onehot_fine-decoder_int-interval-irregular.py {N} {k} {nb_pkts} {length_training}")

# # os.system(f"{program} autoencoder_array-MAP.py {N} {k} {nb_pkts} {length_training}")
# os.system(f"{program} autoencoder_array-MAP-lambda.py {N} {k} {nb_pkts} {length_training}")
# # os.system(f"{program} autoencoder_array-MAP-lambda-soft.py {N} {k} {nb_pkts} {length_training}")


# \Python3\python.exe launcher.py BAC 8 4 4 10000 medium
# \Python3\python.exe launcher.py BAC 4 2 2 10000 bug
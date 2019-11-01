import numpy as np
from os.path import join
import argparse

from utils import seq2onehot

# Pass input and output path names..
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()

# Create path variables.
path_data = args.input_path
path_output = args.output_path

# Load training, validation and test sets.
sequences_tr = np.load(join(path_data, 'sequences_train.npy'))
sequences_val = np.load(join(path_data, 'sequences_validation.npy'))
sequences_tst = np.load(join(path_data, 'sequences_test.npy'))

# Transform the sequences with a one-hot encoding.
sequences_tr_onehot = seq2onehot(sequences_tr, vector_form=False)
sequences_val_onehot = seq2onehot(sequences_val, vector_form=False)
sequences_tst_onehot = seq2onehot(sequences_tst, vector_form=False)

# Save the onehot encoded sequences.
np.save(join(path_output, 'sequences_train_onehot'), sequences_tr_onehot)
np.save(join(path_output, 'sequences_validation_onehot'), sequences_val_onehot)
np.save(join(path_output, 'sequences_test_onehot'), sequences_tst_onehot)


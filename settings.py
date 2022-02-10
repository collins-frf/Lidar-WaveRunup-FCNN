import tensorflow.keras as keras
import tensorflow as tf
import re
import glob
import losses


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

filters = 64
learning_rate = 0.00005
dimensions = 1024
matlength = 12787
noise_std = 0.00
epoch_range = 200
ensemble_runs = 1
batch_size = 10
test_size = 104#104#len(glob.glob('./data/test/features_labels*.npy'))
validation_size = 50
start_row = 0
timestep_plot_indice = 0
row_skip = 512
pred_length = 6144 #512-6144, 256-6272

activation = 'relu'
optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
loss = losses.absolute_error
logs_path = './logs/_noise' + str(learning_rate)

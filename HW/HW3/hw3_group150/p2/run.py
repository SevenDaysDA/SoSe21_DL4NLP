# ------------------------------------------------
#             2.1 Creating Data Splits
# ------------------------------------------------

from keras.preprocessing.sequence import pad_sequences
import random
import os
import tensorflow as tf

################################
input_file = 'data.txt'
################################

tmp_dir = '/tmp'
train_verbose = 1
pad_length = 300

def read_data(input_file):
    vocab = {0}
    data_x = []
    data_y = []
    with open(input_file) as f:
        for line in f:
            label, content = line.split('\t')
            content = [int(v) for v in content.split()]
            vocab.update(content)
            data_x.append(content)
            label = tuple(int(v) for v in label.split())
            data_y.append(label)

    data_x = pad_sequences(data_x, maxlen=pad_length)
    return list(zip(data_y , data_x)), vocab

data, vocab = read_data(input_file)
vocab_size = max(vocab) + 1

# random seeds
random.seed(42)
tf.random.set_seed(42)

random.shuffle(data)
input_len = len(data)

# train_y: a list of 20-component one-hot vectors representing newsgroups
# train_y: a list of 300-component vectors where each entry corresponds to a word ID
train_y, train_x = zip(*(data[:(input_len * 8) // 10]))
dev_y, dev_x = zip(*(data[(input_len * 8) // 10: (input_len * 9) // 10]))
test_y, test_x = zip(*(data[(input_len * 9) // 10:]))




# ------------------------------------------------
#                 2.2 A Basic CNN
# ------------------------------------------------

print("----------------------------------------")
print("Excercise 2.2: Basic CNN")

from keras.models import Sequential, Model
from keras.layers import *

import numpy as np
train_x, train_y = np.array(train_x), np.array(train_y)
dev_x, dev_y = np.array(dev_x), np.array(dev_y)
test_x, test_y = np.array(test_x), np.array(test_y)

# Leave those unmodified and, if requested by the task, modify them locally in the specific task
batch_size = 64
embedding_dims = 100
epochs = 2
filters = 75
kernel_size = 3     # Keras uses a different definition where a kernel size of 3 means that 3 words are convolved at each step


model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=pad_length))

####################################

model.add(Conv1D (filters=filters,kernel_size=2,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(20, activation='softmax'))
print(model.summary())
####################################

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose)
print('Accuracy of simple CNN: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])

print("----------------------------------------")

# ------------------------------------------------
#                2.3 Early Stopping
# ------------------------------------------------

####################################
print("----------------------------------------")
print("Excercise 2.3: Early Stopping")

epochs = 50

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='max',
    save_best_only=True,
    )


callback_earlystoppping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)



model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=train_verbose, callbacks=[model_checkpoint_callback, callback_earlystoppping])
print('Accuracy of Dev Test: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])
print('Accuracy of Test set: %f\n' % model.evaluate(dev_x, dev_y, verbose=0)[1])

####################################
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributed
import json
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def gen_model(hidden_size, vocab_size):
    # simple 2 layer LSTM
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True,
                   input_shape=(None, vocab_size)))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc', 'mse'])
    model.summary()
    return model


with open("projects-winners.json", "r") as f:
    dataset = json.load(f)


def one_hot(vocab_size, i):
    ret = np.zeros(vocab_size)
    ret[i] = 1
    return ret


# first network is just gonna use names, second will predict descriptions from names
name_dataset = [d[1] for d in dataset if d[1]]
chars = []
for name in name_dataset:
    chars += name
chars = set(chars)
chars = {c: i+1 for i, c in enumerate(chars)}
chars['start'] = len(chars)+1
chars['end'] = len(chars)+1
chars['none'] = 0
rev_chars = {i: c for i, c in enumerate(chars)}
with open("chars.json", "w") as f:
    json.dump(chars, f)
for i, name in enumerate(name_dataset):
    name_dataset[i] = [chars['start']] + [chars[c]
                                          for c in name] + [chars['end']]
name_dataset = pad_sequences(
    name_dataset, maxlen=250, padding='post', truncating='post', value=0)
print(name_dataset.shape)
model = gen_model(700, len(chars))


def arr_gen(x, batch_size):
    while True:
        batch_x = []
        batch_y = []
        for name_arr in x:
            batch_x.append([one_hot(len(chars), i)
                            for i in name_arr[:len(name_arr)-1]])
            batch_y.append([one_hot(len(chars), i) for i in name_arr[1:]])

            if len(batch_x) >= batch_size:
                yield (np.array(batch_x), np.array(batch_y))
                batch_x = []
                batch_y = []
        if len(batch_x) > 0:
            yield (np.array(batch_x), np.array(batch_y))
            batch_x = []
            batch_y = []


import math
batch_size = 32
model.fit_generator(
    arr_gen(name_dataset[:math.ceil(len(name_dataset)*0.9)], batch_size), validation_data=arr_gen(name_dataset[math.ceil(len(name_dataset)*0.9):], batch_size),
    steps_per_epoch=math.ceil(len(name_dataset)*0.9/batch_size), validation_steps=math.ceil(len(name_dataset)*0.1/batch_size), epochs=5)

model.save('ideas.model')

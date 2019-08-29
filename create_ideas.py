import numpy as np
import json
from keras.models import load_model
with open("chars.json") as f:
    chars = json.load(f)
inv_chars = {i: c for c, i in chars.items()}

model = load_model('ideas.model')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for j in range(100):
    x_pred = np.zeros((1, 500, len(chars)))
    x_pred[0, 0, chars['start']] = 1
    sent = ""
    for i in range(1, 500):
        preds = model.predict(x_pred, verbose=0)[0][i-1]
        next_index = sample(preds, .5)
        next_char = inv_chars[next_index]
        print(sent)
        if next_index == chars['end']:
            print(sent)
            break

        sent += next_char
        x_pred[0, i, next_index] = 1


# coding: utf-8

# In[1]:

#import tensorflow as tf
import numpy as np
import pandas as pd
import re

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from __future__ import print_function

import random
import sys


# In[2]:

def split_data(df, train_perc = 0.8):
    df['train'] = np.random.rand(len(df)) < train_perc
    train = df[df.train == 1]
    test = df[df.train == 0]
    split_data ={'train': train, 'test': test}
    return split_data

def cleanstr(somestring):
    rx = re.compile('\W+')
    return rx.sub(' ', somestring).strip()


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# In[3]:

df = pd.read_csv('friends-transcripts corpus.txt',delimiter='\t')

#Clean Reading Corpus
df = df[2:]
df.drop("Season & Episode", axis=1 , inplace=True)

#Correcting Dtypes
df.Season = pd.to_numeric(df.Season , errors='raise')
df.Episode = pd.to_numeric(df.Episode, errors='coerce')
df.Episode = df.Episode.replace(np.nan , 17)
df.Title = df.Title.astype(str)
df.Quote = df.Quote.astype(str)
df.Author = df.Author.astype(str)

#c1 = df.Quote[df.Author.str.contains("Rachel")]
#c2 = df.Quote[df.Author.str.contains("Ross")]
#c3 = df.Quote[df.Author.str.contains("Chandler")]
#c4 = df.Quote[df.Author.str.contains("Monica")]
#c5 = df.Quote[df.Author.str.contains("Joey")]
#c6 = df.Quote[df.Author.str.contains("Phoebe")]
#print len(c1), len(c2), len(c3), len(c4), len(c5), len(c6)

#Preliminary Analysis
print ("Who talks how much ! \n")
print (df.Author.value_counts()[0:6])

#Sampling Dataset

#dict of Dataframes
Dataset = split_data(df , train_perc=0.8)
print ("Total rows   :" , df.shape[0])
print ("Training set :" , len(Dataset['train']))
print ("Test Set     :" , len(Dataset['test']))


# In[25]:

#Takes one Hour to Train!

from keras.models import model_from_json

friend = "Ross"
train=False
save=True


text = ' '.join(Dataset['train'].Quote[Dataset['train'].Author == friend].tolist())
text = text.lower()
print('corpus length:', len(text))

chars = set(text)

if not chars:
    print ("Invalid friends character, type: Ross Rachel Phoebe Chandler Monica or Joey")
    sys.exit(1)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1



# build the model: 2 stacked LSTM
if train:
    print('Build model...')
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
else:
    print ("Attempting to load model from h5py")
    model = model_from_json(open(friend+'.json').read())
    model.load_weights(friend+'.h5')

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# train the model, output generated text after each iteration

for iteration in range(1, 40):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    if save:
        json_string = model.to_json()
        open(friend+'.json', 'w').write(json_string)
        model.save_weights(friend+'.h5' , overwrite = True)
        print ("Model and weights saved to directory")

    else:
        print ("model not saved")


# In[5]:

#generateSentencesLike("Phoebe" , train=True , save=True)
#Takes one Hour to Train!

#Rachel 
#Chandle 
#Phoebe


# In[ ]:

#generateSentencesLike("Rachel" , train=True , save=True)


# In[6]:

#model = model_from_json(open(friend+'.json').read())
#model.load_weights(friend+'.h5')


# In[24]:

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#model.fit(X, y, batch_size=128, nb_epoch=1)
#model.evaluate(X,y, batch_size=128 )
    


# In[20]:

#for i in range(0,15):
#    print (model.get_weights()[i].shape)


# In[ ]:

print model.


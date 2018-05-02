import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
#import tensorflow

import sys
import os

#os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, History
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, merge, concatenate
from keras.layers import Dense, Input, Flatten, Concatenate, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

data_train = pd.read_csv('/home/anavaleria/ubuntu/faq_preprocessing/QA_data/QQ_train.csv', sep=',')
print data_train.shape

data_dev = pd.read_csv('/home/anavaleria/ubuntu/faq_preprocessing/QA_data/QQ_dev.csv', sep=',')
print data_dev.shape


data_test = pd.read_csv('/home/anavaleria/ubuntu/faq_preprocessing/QA_data/QQ_test.csv', sep=',')
print data_test.shape

texts = []
labels = []
orgs = []

train_samples = len(data_train)
dev_samples = len(data_dev)

all_data = pd.concat([data_train, data_dev, data_test], axis = 0)
print(all_data.shape)
for idx in range(all_data.RELQ_text.shape[0]):
    org =  all_data.ORGQ_TEXT.tolist()[idx]
    text = all_data.RELQ_text.tolist()[idx]
    texts.append(clean_str(text))
    labels.append(all_data.RELQ_RELEVANCE2ORGQ.tolist()[idx])
    orgs.append(clean_str(org))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts+orgs)
sequences = tokenizer.texts_to_sequences(texts)
sequences_org = tokenizer.texts_to_sequences(orgs)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data_org = pad_sequences(sequences_org, maxlen=MAX_SEQUENCE_LENGTH)

for i in range(len(labels)):
    if labels[i] == 'PerfectMatch':
        labels[i] = 'Relevant'
bin_labels = []
for i in labels:
    if i== 'Relevant':
        bin_labels.append(0)
    else:
        bin_labels.append(1)

print(set(labels))
labels = to_categorical(np.asarray(bin_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


'''
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
'''

x_train = data[0:train_samples]
y_train = labels[0:train_samples]

org_train = data_org[0:train_samples]
org_val = data_org[train_samples:train_samples+dev_samples]
org_test = data_org[train_samples+dev_samples::]


x_val = data[train_samples:train_samples+dev_samples]  #change to val data
y_val = labels[train_samples:train_samples+dev_samples]

print('Traing and validation set number of positive and negative reviews')
print y_train.sum(axis=0)
print y_val.sum(axis=0)

GLOVE_DIR = "/home/anavaleria/ubuntu/faq_preprocessing/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))


# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel',
                                      shape=(input_shape[-1],),
                                      initializer='normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)   # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

filepath="best.hdf5"

import os.path
if os.path.exists(filepath) == True:

    print('Building from saved model...')
    bestmodel =  load_model(filepath)
    assert_allclose(model.predict(x_train),
                    bestmodel.predict(x_train),
                    1e-5)

    # fit the model
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history = History()
    callbacks_list = [checkpoint, history]
    bestmodel.summary()
    bestmodel.fit([x_train, org_train], y_train,
                validation_data=([x_val,org_val], y_val),
                nb_epoch=10, batch_size=128, verbose = 1,
                callbacks = [checkpoint])
else:

    print('No saved model. Building New model...')
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    embedding_layer2 = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)



    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_gru = Bidirectional(GRU(300, return_sequences=True))(embedded_sequences)
    norm1 = Dropout(0.5)(l_gru)
    l_gru11 = Bidirectional(GRU(600, return_sequences=True))(norm1)
    norm11 = Dropout(0.5)(l_gru)
    #l_att = AttLayer()(l_gru)

    sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences2 = embedding_layer2(sequence_input2)
    l_gru2 = Bidirectional(GRU(300, return_sequences=True))(embedded_sequences2)
    norm2 = Dropout(0.5)(l_gru2)
    l_gru22 = Bidirectional(GRU(600, return_sequences=True))(norm2)
    norm22 = Dropout(0.5)(l_gru22)
    #l_att2 = AttLayer()(l_gru2)



    merged = concatenate([norm11, norm22])
    l_gru3 = Bidirectional(GRU(2400, return_sequences=False))(merged)



    preds = Dense(2, activation='softmax')(l_gru3)
    model = Model(inputs = [sequence_input,sequence_input2], outputs = [preds])
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

    history = History()
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #                            save_weights_only=True)


    print("model fitting - BiGRU network")
    model.summary()
    model.fit([x_train, org_train], y_train,
                validation_data=([x_val,org_val], y_val),
                nb_epoch=10, batch_size=128, verbose = 1,
                callbacks = [checkpoint, history])

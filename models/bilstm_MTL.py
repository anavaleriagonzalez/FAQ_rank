import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer

train = pd.read_csv('QQ_train.csv', sep = ',')
dev = pd.read_csv('QQ_dev.csv', sep = ',')
test = pd.read_csv('QQ_test-17.csv', sep = ',')

indices = [78, 102, 2366, 3718]

auxtrain = pd.read_csv('QA_train-1.csv', sep = ',')

auxdev = pd.read_csv('QA_dev-1.csv', sep = ',')
auxtest = pd.read_csv('QA_test-1.csv', sep = ',')

auxtrain = auxtrain.dropna(axis = 0)[0:4000]
auxdev = auxdev.dropna(axis = 0)[0:1000]
auxtest = auxtest.dropna(axis = 0)[0:1000]


auxtrain = pd.concat([auxtrain[0:indices[0]-1], auxtrain[indices[0]+1:indices[1]-1], auxtrain[indices[1]+1:indices[2]-1],
                      auxtrain[indices[2]+1:indices[3]-1], auxtrain[indices[3]+1::]])


MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

texts = []
labels = []
orgs = []

train_samples = len(train)
dev_samples = len(dev)

all_data = pd.concat([train, dev,test], axis = 0)
print(all_data.shape)

for idx in range(all_data.RELQ_text.shape[0]):
    org =  all_data.ORGQ_TEXT.tolist()[idx]
    text = all_data.RELQ_text.tolist()[idx]

    texts.append(text.lower())
    labels.append(all_data.RELQ_RELEVANCE2ORGQ.tolist()[idx])
    orgs.append(org.lower())

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts+orgs)
sequences = tokenizer.texts_to_sequences(texts)
sequences_org = tokenizer.texts_to_sequences(orgs)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data_org = pad_sequences(sequences_org, maxlen=MAX_SEQUENCE_LENGTH)


#REPEAT PROCESS FOR AUX DATA
texts = []
labels_aux = []
orgs = []

train_samplesaux = len(auxtrain)
dev_samplesaux = len(auxdev)

all_data_aux = pd.concat([auxtrain, auxdev,auxtest], axis = 0)
print(all_data_aux.shape)

for idx in range(all_data_aux.RELQ_TEXT.shape[0]):
    ANS =  all_data_aux.RELC_text.tolist()[idx]
    text = all_data_aux.RELQ_TEXT.tolist()[idx]

    texts.append(text.lower())
    labels_aux.append(all_data_aux.RELC_RELEVANCE2RELQ.tolist()[idx])
    orgs.append(ANS.lower())

tokenizer2 = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer2.fit_on_texts(texts+orgs)
sequences2 = tokenizer2.texts_to_sequences(texts)
sequences_org2 = tokenizer2.texts_to_sequences(orgs)

word_index_AUX = tokenizer2.word_index
print('Found %s unique tokens.' % len(word_index_AUX))

data_aux = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH)
data_ans_aux = pad_sequences(sequences_org2, maxlen=MAX_SEQUENCE_LENGTH)

print('loading GLOVE embeddings....')
embeddings_path = '/home/ana/Desktop/Current_projects/FAQ_rank/glove.6B.50d.txt'


embeddings_index = {}
f = open(embeddings_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#ALL_MAIN DATA
x_train = data[0:train_samples]
y_train = labels[0:train_samples]

org_train = data_org[0:train_samples]
org_val = data_org[train_samples:train_samples+dev_samples]
org_test = data_org[train_samples+dev_samples::]


x_val = data[train_samples:train_samples+dev_samples]  #change to val data
y_val = labels[train_samples:train_samples+dev_samples]

x_test = data[train_samples+dev_samples::]  #change to val data
y_test = labels[train_samples+dev_samples::]

f = open('qq_data-17.p', 'rb')
datamain = pickle.load(f)
f.close()

x_train_sim = datamain[0]
x_dev_sim= datamain[1]
x_test_sim =datamain[2]
tr_labels = datamain[3]
dev_labels = datamain[4]
test_labels = datamain[5]


#ALL AUX DATA


x_trainAUX = data_aux[0:train_samplesaux]
y_trainAUX = labels_aux[0:train_samplesaux]

org_trainAUX = data_ans_aux[0:train_samplesaux]


org_valAUX = data_ans_aux[train_samplesaux:train_samplesaux+dev_samplesaux]
org_testAUX = data_ans_aux[train_samplesaux+dev_samplesaux::]


x_valAUX = data_aux[train_samplesaux:train_samplesaux+dev_samplesaux]  #change to val data
y_valAUX = labels_aux[train_samplesaux:train_samplesaux+dev_samplesaux]

x_testAUX = data_aux[train_samplesaux+dev_samplesaux::]  #change to val data
y_testAUX = labels_aux[train_samplesaux+dev_samplesaux::]

f = open('QA_data-1.p', 'rb')
data_aux = pickle.load(f)
f.close()

AUX_train = data_aux[0]
AUX_dev = data_aux[1]
AUX_test  = np.concatenate((data_aux[2], data_aux[2]), axis = 0)
tr_labelsAUX = data_aux[3]
dev_labelsAUX = data_aux[4]
test_labelsAUX = data_aux[5] + data_aux[5]

print(len(test_labelsAUX))


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_matrix2 = np.random.random((len(word_index_AUX) + 1, EMBEDDING_DIM))
for word, i in word_index_AUX.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix2[i] = embedding_vector
##embeddings look up
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

embedding_layer2 = Embedding(len(word_index_AUX) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix2],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

print('building model...')

from keras.models import Sequential , load_model, Model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import SGD
from keras.layers import Embedding, merge, concatenate
from keras.layers import Dense, Input, Flatten, Concatenate, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
import pickle
import pandas as pd
import numpy as np
filepath="MAP FILES/MTL-BILSTM-QQQA-17.hdf5"

'''
#MAIN INPUT1
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm1 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences)
norm1 = Dropout(0.5)(lstm1)

#MAIN INPUT2
sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences2 = embedding_layer(sequence_input2)
lstm2 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences2)
norm2 = Dropout(0.5)(lstm2)

#AUX INPUT1
sequence_input3 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences3 = embedding_layer2(sequence_input3)
lstm3 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences3)
norm3 = Dropout(0.5)(lstm3)

#AUX INPUT2
sequence_input4 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences4 = embedding_layer2(sequence_input4)
lstm4 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences4)
norm4 = Dropout(0.5)(lstm4)


#main similarity input
sim_input1 =  Input(shape=(20,), dtype='float32')
merge1 = concatenate([norm1, norm2, sim_input1])

#main similarity input
sim_input2 =  Input(shape=(20,), dtype='float32')
merge2 = concatenate([norm3, norm4, sim_input2])

#shared layer
shared_layer  = Dense(64,  activation='relu')
shared_out1 = shared_layer(merge1)
drop1 = Dropout(0.5)(shared_out1)

shared_out2 = shared_layer(merge2)
drop2 = Dropout(0.5)(shared_out2)

#TASK SPECIFIC LAYERS

#main
main2 = Dense(64, activation='relu')(drop1)
drop3 = Dropout(0.5)(main2)


#aux
aux2 = Dense(64, activation='relu')(drop2)
drop4 = Dropout(0.5)(aux2)



#main
main3 = Dense(64, activation='relu')(drop3)
drop5 = Dropout(0.5)(main3)


#aux
aux3 = Dense(64, activation='relu')(drop4)
drop6 = Dropout(0.5)(aux3)

out_main = Dense(1, activation='sigmoid', name='main')(drop5)

out_aux =  Dense(1, activation='sigmoid')(drop6)



batch_size = 100
nb_epoch = 1000

model = Model(inputs=[sequence_input, sequence_input2, sim_input1 ,sequence_input3, sequence_input4, sim_input2],
              outputs=[out_main, out_aux])

sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=['binary_crossentropy','binary_crossentropy'],
              metrics=['accuracy'],
             loss_weights=[0.8, 0.05])
print(model.summary())


model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='val_main_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit([np.array(x_train),np.array(org_train), np.array(x_train_sim),
                     np.array(x_trainAUX[0:len(x_train)]),np.array(org_trainAUX[0:len(x_train)]),
                     np.array(AUX_train[0:len(x_train)])],
                    [np.array(tr_labels), np.array(tr_labelsAUX[0:len(x_train)])],
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,verbose=1,
                    validation_data= ([np.array(x_val),np.array(org_val), np.array(x_dev_sim),
                     np.array(x_valAUX[0:len(x_val)]),np.array(org_valAUX[0:len(x_val)]),
                     np.array(AUX_dev[0:len(x_val)])],
                        [np.array(dev_labels), np.array(dev_labelsAUX[0:len(x_val)])]),
                    callbacks = callbacks_list)
'''
best_model = load_model(filepath)
score = best_model.evaluate([np.array(x_test),np.array(org_test), np.array(x_test_sim),
                     np.array(x_testAUX[0:len(x_test)]),np.array(org_testAUX[0:len(x_test)]),
                     np.array(AUX_test[0:len(x_test)])],
                            [np.array(test_labels), np.array(test_labelsAUX[0:len(x_test)])], verbose=0)
print(best_model.metrics_names)

print('Test accuracy on main task:', score)




probs = best_model.predict([np.array(x_test),np.array(org_test), np.array(x_test_sim),
                     np.array(x_testAUX[0:len(x_test)]),np.array(org_testAUX[0:len(x_test)]),
                     np.array(AUX_test[0:len(x_test)])])
probs_list = [i[0] for i in probs[0]]



test = pd.read_csv('QQ_test-17.csv', sep = ',')
#GENERATE FILE FOR MAP EVALUATION SCRIPT
with open('MTL-BILSTM-QQQA-17.preds', 'w') as f:
    for i in range(len(probs_list)):

        label = 'false'
        #'\t'+str(entry['probs'].tolist()[0])

        f.write(test.RELQ_ID.tolist()[i]+ '\t'+test['RELC_ID'].tolist()[i]+'\t'+ str(i)+ '\t'+str(probs_list[i])+'\t'+  label +'\n')

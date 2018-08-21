import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
filepath="models/best2_stl_LSTM-MLP.hdf5"
train = pd.read_csv('QQ_train.csv', sep = ',')
dev = pd.read_csv('QQ_dev.csv', sep = ',')
test = pd.read_csv('QQ_test.csv', sep = ',')


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

print('loading GLOVE embeddings....')
#embeddings_path = '/home/ana/Desktop/Current_projects/FAQ_rank/glove.6B.50d.txt'
embeddings_path = '/home/ana/Desktop/Current_projects/QA_data/data_preprocessing/data_dumps/glove.6B.50d.txt'


embeddings_index = {}
f = open(embeddings_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


x_train = data[0:train_samples]
y_train = labels[0:train_samples]

org_train = data_org[0:train_samples]
org_val = data_org[train_samples:train_samples+dev_samples]
org_test = data_org[train_samples+dev_samples::]


x_val = data[train_samples:train_samples+dev_samples]  #change to val data
y_val = labels[train_samples:train_samples+dev_samples]

x_test = data[train_samples+dev_samples::]  #change to val data
y_test = labels[train_samples+dev_samples::]


print('loading similarity features...')
f = open('qq_data.p', 'rb')
datamain = pickle.load(f)
f.close()

x_train_sim = datamain[0]
x_dev_sim= datamain[1]
x_test_sim =datamain[2]
tr_labels = datamain[3]
dev_labels = datamain[4]
test_labels = datamain[5]


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

##embeddings look up
from keras.layers import Embedding
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

from keras.models import Sequential , load_model, Model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import SGD
from keras.layers import Embedding, merge, concatenate
from keras.layers import Dense, Input, Flatten, Concatenate, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.callbacks import ModelCheckpoint
import pickle
import pandas as pd


print('building model...')
'''
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm1 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences)
norm1 = Dropout(0.5)(lstm1)


#lstm11 = Bidirectional(LSTM(100, return_sequences=False))(norm1)
##norm11 = Dropout(0.5)(lstm11)


sequence_input2 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences2 = embedding_layer2(sequence_input2)
lstm2 = Bidirectional(LSTM(50, return_sequences=False))(embedded_sequences2)
norm2 = Dropout(0.5)(lstm2)


##lstm22 = Bidirectional(LSTM(100, return_sequences=False))(norm2)
##norm22 = Dropout(0.5)(lstm22)


sim_input =  Input(shape=(20,), dtype='float32')
merge = concatenate([norm1, norm2, sim_input ])

dense1 = Dense(64,  activation='relu')(merge)
drop1 = Dropout(0.5)(dense1)

dense2 = Dense(64, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)

dense3 = Dense(64,  activation='relu')(drop2)
drop3 = Dropout(0.5)(dense3)

out = Dense(1, activation='sigmoid')(drop3)


batch_size = 100
nb_epoch = 3000

model = Model(inputs=[sequence_input, sequence_input2, sim_input], outputs=[out])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=['binary_crossentropy'],
              metrics=['accuracy'])
print(model.summary())


#model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('training....')
history = model.fit([x_train, org_train, np.array(x_train_sim)], np.array(tr_labels), batch_size=batch_size,
                    nb_epoch=nb_epoch,verbose=1,
                    validation_data=([x_val, org_val, np.array(x_dev_sim)], [np.array(dev_labels)]),
                   callbacks = callbacks_list)
'''


def predict_classes(probs):
    '''Generate class predictions for the input samples
    batch by batch.
    # Arguments
        x: input data, as a Numpy array or list of Numpy arrays
            (if the model has multiple inputs).
        batch_size: integer.
        verbose: verbosity mode, 0 or 1.
    # Returns
        A numpy array of class predictions.
    '''
    proba = []
    for probab in probs:
        proba.append((probab > 0.5).astype('int32'))
    return proba

print('evaluating best model...')
best_model = load_model(filepath)
score = best_model.evaluate(x = [x_test, org_test,  x_test_sim],y = np.array(test_labels), verbose=0)

print('Test accuracy:', score[1])


probs = best_model.predict([x_test, org_test,  x_test_sim])
probs_list = [i[0] for i in probs]

prob_file = open('PREDS/probs-stl-bilstm-16', 'wb')
pickle.dump(probs_list, prob_file)
prob_file.close()

predictions = predict_classes(np.array(probs))
preds_list = [i[0] for i in predictions]
print(preds_list)

pred_file = open('PREDS/preds-stl-bilstm-16', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()
'''
print('saving predictions to text file....')
with open('PREDS/bilstm_stl-16.preds', 'w') as f:
    for i in range(len(probs_list)):

        if probs_list[i] < .52:
            label = 'false'
        else:
            label = 'true'
        #'\t'+str(entry['probs'].tolist()[0])

        f.write(test.ORGQ_ID.tolist()[i]+ '\t'+test['RELQ_ID'].tolist()[i]+'\t'+ str(i)+ '\t'+str(probs_list[i])+'\t'+  label +'\n')
'''

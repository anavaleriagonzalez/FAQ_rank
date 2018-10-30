from keras.models import Sequential , load_model
from keras.layers import Dense, Activation , Dropout,BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import pickle

import pandas as pd
import numpy as np
import random
import argparse
random.seed(10)


parser = argparse.ArgumentParser(description='Arguments for mlp model')
parser.add_argument('model_name',  type=str,
                     help='Provide the name of the model to be saved')
parser.add_argument('query_path',  type=str,
                     help='Provide the full path to the QQ dataframes')  

args = parser.parse_args()  

filepath="../trained_models/"+args.model_name+'.hdf5'


f = open(args.query_path, 'rb')
datamain = pickle.load(f)
f.close()

x_train = datamain[0]
print( len(x_train))

x_train[np.argwhere(np.isnan(x_train))] = 0
print('now', x_train[0].shape)

x_dev= datamain[1] 

x_dev[np.argwhere(np.isnan(x_dev))] = 0
print(np.argwhere(np.isnan(x_dev)))

x_test =datamain[2]
print(x_test.shape)
x_test[np.argwhere(np.isnan(x_test))] = 0
print(np.argwhere(np.isnan(x_test)))

from keras.utils.np_utils import to_categorical
from keras import backend as K
import itertools
class WeightedCategoricalCrossEntropy(object):
    
    def __init__(self, weights):
        nb_cl = len(weights)
        self.weights = np.ones((nb_cl, nb_cl))
        for class_idx, class_weight in weights.items():
            self.weights[0][class_idx] = class_weight
            self.weights[class_idx][0] = class_weight
        self.__name__ = 'w_categorical_crossentropy'
    
    def __call__(self, y_true, y_pred):
        return self.w_categorical_crossentropy(y_true, y_pred)
        
    def w_categorical_crossentropy(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[..., 0])
        y_pred_max = K.max(y_pred, axis=-1)
        y_pred_max = K.expand_dims(y_pred_max, axis=-1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
            w = K.cast(self.weights[c_t, c_p], K.floatx())
            y_p = K.cast(y_pred_max_mat[..., c_p], K.floatx())
            y_t = K.cast(y_pred_max_mat[..., c_t], K.floatx())
            final_mask += w * y_p * y_t
        return K.categorical_crossentropy(y_pred, y_true) * final_mask


import functools
from functools import partial
from keras.constraints import max_norm
#ncce = WeightedCategoricalCrossEntropy({0:.1, 1:.3})

tr_labels = to_categorical(datamain[3])
dev_labels = to_categorical(datamain[4] )
test_labels = to_categorical(datamain[5])
'''
tr_labels = datamain[3]
dev_labels = datamain[4]
test_labels = datamain[5]
'''

print('here',x_test[0])
####Building STL MODEL#######
from sklearn import preprocessing

x_1 = [item[0] for item in x_test]
x_train = preprocessing.normalize(x_train)
x_dev = preprocessing.normalize(x_dev)
x_test = preprocessing.normalize(x_test)



print(len(x_test[0]))
model = Sequential()
model.add(BatchNormalization())
model.add(Dense(150, input_dim=len(x_test[0]), kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(1)))
model.add(Dropout(0.1))



model.add(Dense(150, input_dim=len(x_test[0]), kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(1)))
model.add(Dropout(0.2))



model.add(Dense(150, input_dim=len(x_test[0]), kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(1)))
model.add(Dropout(0.2))


model.add(Dense(150, input_dim=len(x_test[0]), activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(2, activation='softmax'))
batch_size = 60
nb_epoch =500


adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


#model.load_weights(filepath)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit(np.array(x_train), np.array(tr_labels), batch_size=batch_size,
                    nb_epoch=nb_epoch,verbose=1,
                    validation_data=([np.array(x_dev)], [np.array(dev_labels)]),
                   callbacks = callbacks_list)



print('using best model for predictions')
best_model = load_model(filepath)
score = best_model.evaluate(np.array(x_test), np.array(test_labels), verbose=0)

print('Test accuracy:', score[1])

predictions = best_model.predict_classes(x_test)
print(predictions)
#preds_list = [i[0] for i in predictions]

probs = best_model.predict(x_test)
print(probs)
probs_list = probs

prob_file = open('../outputs/probs_'+args.model_name+'.p','wb')
pickle.dump(probs_list, prob_file)
prob_file.close()

#predictions = best_model.predict_classes(x_test)
preds_list = predictions

pred_file = open('../outputs/preds_'+args.model_name+'.p', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()

#SANITY CHECK

correct = 0
for i in range(len(preds_list)):
    if preds_list[i] == datamain[5][i]:
        correct = correct + 1

print('double checking accuracy...')
print(correct / len(predictions))

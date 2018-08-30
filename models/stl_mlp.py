from keras.models import Sequential , load_model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
import pandas as pd
import numpy as np
import random
import argparse
random.seed(7)


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
print(x_train[0])

x_train[np.argwhere(np.isnan(x_train))] = 0
print(x_train[0].shape)

x_dev= datamain[1] 
print(np.argwhere(np.isnan(x_dev)))

x_test =datamain[2]
print(x_test.shape)
print(np.argwhere(np.isnan(x_test)))

tr_labels = datamain[3]
dev_labels = datamain[4] 
test_labels = datamain[5]

print(x_test[0])
####Building STL MODEL#######
from sklearn import preprocessing

x_train = preprocessing.normalize(x_train)
x_dev = preprocessing.normalize(x_dev)
x_test = preprocessing.normalize(x_test)

print(len(x_test[0]))
model = Sequential()
model.add(Dense(150, input_dim=len(x_test[0]), activation='relu'))
model.add(Dropout(0.0))

model.add(Dense(150, input_dim=len(x_test[0]), activation='relu'))
model.add(Dropout(0.0))

model.add(Dense(150, input_dim=len(x_test[0]), activation='relu'))
model.add(Dropout(0.0))

model.add(Dense(1, activation='sigmoid'))
batch_size = 500
nb_epoch =500


sgd = SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


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
preds_list = [i[0] for i in predictions]

probs = best_model.predict(x_test)
probs_list = [i[0] for i in probs]

prob_file = open('../outputs/probs_'+args.model_name+'.p','wb')
pickle.dump(probs_list, prob_file)
prob_file.close()

predictions = best_model.predict_classes(x_test)
preds_list = [i[0] for i in predictions]

pred_file = open('../outputs/preds_'+args.model_name+'.p', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()

#SANITY CHECK

correct = 0
for i in range(len(preds_list)):
    if preds_list[i] == test_labels[i]:
        correct = correct + 1

print('double checking accuracy...')
print(correct / len(predictions))



test = pd.read_csv('../QA_data/data_dumps/QQ_test-17.csv', sep = ',')


#GENERATE FILE FOR MAP EVALUATION SCRIPT
with open('../MAP/Predicted/MTL-STL-17.preds', 'w') as f:
    for i in range(len(probs_list)):

        if preds_list[i] == 1:
            label = 'true'
        else:
            label = 'false'
        #'\t'+str(entry['probs'].tolist()[0])

        f.write(test.ORGQ_ID.tolist()[i]+ '\t'+test['RELQ_ID'].tolist()[i]+'\t'+ str(i)+ '\t'+str(probs_list[i])+'\t'+  label +'\n')


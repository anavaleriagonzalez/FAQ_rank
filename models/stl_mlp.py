from keras.models import Sequential , load_model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
import pandas as pd
import numpy as np
import random
filepath="models/best2_stl_mlp.hdf5"

rands = [i for i in range(26000)]
random.shuffle(rands)
print(rands)
f = open('qq_data-17.p', 'rb')
datamain = pickle.load(f)
f.close()

rands2 = [i for i in range(6300)]
random.shuffle(rands2)

x_train = datamain[0]#np.concatenate((datamain[1], datamain[1]), axis = 0)# np.array([datamain[0][i] for i in rands])
# np.concatenate((i[0:6], i[-6::]),axis = 0) for bow+emb
#x_train = np.array([np.concatenate((i[0:12], i[13::]),axis = 0) for i in x_train])
x_dev= datamain[1] #np.array([datamain[1][i] for i in rands2])
#print(x_train[0].shape)
#x_dev = np.array([np.concatenate((i[0:12], i[13::]),axis = 0) for i in x_dev])

x_test =datamain[2]
print(x_test.shape)

#x_test = np.array([np.concatenate((i[0:12], i[13::]),axis = 0)  for i in x_test])
tr_labels = datamain[3]#np.concatenate((datamain[4] , datamain[4]) , axis = 0) #np.array([datamain[3][i] for i in rands])
dev_labels = datamain[4] #np.array([datamain[4][i] for i in rands2])
test_labels = datamain[5]

print(x_test[0])
####STL MODEL#######

len(x_train[0])
model = Sequential()
model.add(Dense(150, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(150, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(150, input_dim=len(x_train[0]), activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
batch_size = 500
nb_epoch =500


sgd = SGD(lr=0.05, decay=1e-8, momentum=0.9, nesterov=True)
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

prob_file = open('PREDS/probs-stl-mlp-17', 'wb')
pickle.dump(probs_list, prob_file)
prob_file.close()

predictions = best_model.predict_classes(x_test)
preds_list = [i[0] for i in predictions]

pred_file = open('PREDS/preds-stl-mlp-17', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()

#SANITY CHECK

correct = 0
for i in range(len(preds_list)):
    if preds_list[i] == test_labels[i]:
        correct = correct + 1

print('double checking accuracy...')
print(correct / len(predictions))



#test = pd.read_csv('QA_test-2.csv', sep = ',')

'''
#GENERATE FILE FOR MAP EVALUATION SCRIPT
with open('ablation/QA-MAIN/STL.preds', 'w') as f:
    for i in range(len(probs_list)):

        if preds_list[i] == 1:
            label = 'true'
        else:
            label = 'false'
        #'\t'+str(entry['probs'].tolist()[0])

        f.write(test.RELQ_ID.tolist()[i]+ '\t'+test['RELC_ID'].tolist()[i]+'\t'+ str(i)+ '\t'+str(probs_list[i])+'\t'+  label +'\n')
'''

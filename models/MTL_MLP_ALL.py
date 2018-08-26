###all tasks ######

from keras.models import Sequential , load_model, Model
from keras.layers import Dense, Activation , Dropout, Input
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
import pandas as pd

filepath="models/best2_mtl_mlp-16-ALL.hdf5"
f = open('qq_data.p', 'rb')
datamain = pickle.load(f)
f.close()

x_train = datamain[0]
x_dev= datamain[1]
x_test =datamain[2]
tr_labels = datamain[3]
dev_labels = datamain[4]
test_labels = datamain[5]

f = open('QA_data-1.p', 'rb')
data_aux = pickle.load(f)
f.close()

AUX_train1 = data_aux[0]
AUX_dev1 = data_aux[1]
AUX_test1  = data_aux[2]
tr_labelsAUX1 = data_aux[3]
dev_labelsAUX1 = data_aux[4]
test_labelsAUX1 = data_aux[5]


f = open('NLI_data.p', 'rb')
data_aux = pickle.load(f)
f.close()

AUX_train2 = data_aux[0]
AUX_dev2 = data_aux[1]
AUX_test2  = np.concatenate((data_aux[2],data_aux[2]),axis = 0)
tr_labelsAUX2 = data_aux[3]
dev_labelsAUX2 = data_aux[4]
test_labelsAUX2 = data_aux[5] + data_aux[5]

f = open('fnc-data.p', 'rb')
data_aux = pickle.load(f)
f.close()

AUX_train3 = data_aux[0]
AUX_dev3 = data_aux[1]
AUX_test3  = data_aux[2]
tr_labelsAUX3 = data_aux[3]
dev_labelsAUX3 = data_aux[4]
test_labelsAUX3 = data_aux[5]
####STL MODEL#######

input1 = Input(shape=(len(x_train[0]),))
input2 = Input(shape=(len(x_train[0]),))
input3 = Input(shape=(len(x_train[0]),))
input4 = Input(shape=(len(x_train[0]),))



main_layer1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.5)(main_layer1)

aux_layer1 = Dense(64, activation='relu')(input2)
drop2 = Dropout(0.5)(aux_layer1)

aux_layer2 = Dense(64, activation='relu')(input3)
drop3 = Dropout(0.5)(aux_layer2)

aux_layer3 = Dense(64, activation='relu')(input4)
drop4 = Dropout(0.5)(aux_layer3)

# a layer instance is callable on a tensor, and returns a tensor
shared_layer= Dense(64, activation='relu')
shared_out1 = shared_layer(drop1)
shared_out2 = shared_layer(drop2)
shared_out3 = shared_layer(drop3)
shared_out4 = shared_layer(drop4)


drop5 = Dropout(0.5)(shared_out1)
drop6 = Dropout(0.5)(shared_out2)

drop7 = Dropout(0.5)(shared_out3)
drop8 = Dropout(0.5)(shared_out4)


main_layer2 = Dense(64, activation='relu')(drop5)
drop9 = Dropout(0.5)(main_layer2)



aux_layer4 = Dense(64, activation='relu')(drop6)
drop10 = Dropout(0.5)(aux_layer4)

aux_layer5 = Dense(64, activation='relu')(drop7)
drop11 = Dropout(0.5)(aux_layer5)

aux_layer6 = Dense(64, activation='relu')(drop8)
drop12 = Dropout(0.5)(aux_layer6)


main_out = Dense(1, activation='sigmoid', name="main")(drop9)
aux_out1 = Dense(1, activation='sigmoid')(drop10)
aux_out2 = Dense(1, activation='sigmoid')(drop11)
aux_out3 = Dense(1, activation='sigmoid')(drop12)


batch_size = 100
nb_epoch = 3000

model = Model(inputs=[input1, input2, input3, input4], outputs=[main_out, aux_out1, aux_out2, aux_out3])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss=['binary_crossentropy','binary_crossentropy','binary_crossentropy','binary_crossentropy'],
              metrics=['accuracy'],
             loss_weights = [.9, .08,.1,.1])


print(model.summary())


checkpoint = ModelCheckpoint(filepath, monitor='val_main_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit([np.array(x_train), np.array(AUX_train1)[0:len(x_train)],np.array(AUX_train2)[0:len(x_train)],np.array(AUX_train3)[0:len(x_train)]],
                    [np.array(tr_labels), np.array(tr_labelsAUX1[0:len(x_train)]),np.array(tr_labelsAUX2[0:len(x_train)]),np.array(tr_labelsAUX3[0:len(x_train)])],
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,verbose=1,
                    validation_data= ([np.array(x_dev), np.array(AUX_dev1[0:len(x_dev)]), np.array(AUX_dev2[0:len(x_dev)]), np.array(AUX_dev3[0:len(x_dev)])],
                                      [np.array(dev_labels), np.array(dev_labelsAUX1[0:len(x_dev)]),np.array(dev_labelsAUX2[0:len(x_dev)]),np.array(dev_labelsAUX3[0:len(x_dev)])]),
                    callbacks = callbacks_list)


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



best_model = load_model(filepath)
score = best_model.evaluate([np.array(x_test), np.array(AUX_test1[0:len(x_test)]),np.array(AUX_test2[0:len(x_test)]),np.array(AUX_test3[0:len(x_test)])],
                            [np.array(test_labels), np.array(test_labelsAUX1[0:len(x_test)]), np.array(test_labelsAUX2[0:len(x_test)]), np.array(test_labelsAUX3[0:len(x_test)])], verbose=0)
print(best_model.metrics_names)
print('Test accuracy on main task:', score)




probs = best_model.predict([x_test,  np.array(AUX_test1[0:len(x_test)]), np.array(AUX_test2[0:len(x_test)]), np.array(AUX_test3[0:len(x_test)])])
probs_list = [i[0] for i in probs[0]]

predictions = predict_classes(np.array(probs))
preds_list = [i[0] for i in predictions[0]]
print(preds_list)

prob_file = open('PREDS/probs-mtl-mlp-16-all', 'wb')
pickle.dump(probs_list, prob_file)
prob_file.close()



pred_file = open('PREDS/preds-mtl-mlp-16-all', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()


'''
test = pd.read_csv('QQ_test.csv', sep = ',')
#GENERATE FILE FOR MAP EVALUATION SCRIPT
with open('test_MTL_MLP-17-ALL.preds', 'w') as f:
    for i in range(len(probs_list)):

        label = 'false'
        #'\t'+str(entry['probs'].tolist()[0])

        f.write(test.ORGQ_ID.tolist()[i]+ '\t'+test['RELQ_ID'].tolist()[i]+'\t'+ str(i)+ '\t'+str(probs_list[i])+'\t'+  label +'\n')
'''

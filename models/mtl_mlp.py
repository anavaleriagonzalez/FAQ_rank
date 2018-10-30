from keras.models import Sequential , load_model, Model
from keras.layers import Dense, Activation , Dropout, Input
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, History
import pickle
import numpy as np
import pandas as pd

filepath="models/best2_mtl_mlp-16-NLI.hdf5"
f = open('qq_data-17.p', 'rb')
datamain = pickle.load(f)
f.close()

x_train = datamain[0]
# np.concatenate((i[0:6], i[-6::]),axis = 0) for bow+emb
#x_train = np.array([np.concatenate((i[0:6], i[-6::]),axis = 0) for i in x_train])
x_dev= datamain[1]

#x_dev = np.array([np.concatenate((i[0:6], i[-6::]),axis = 0) for i in x_dev])

x_test =datamain[2]
print(x_train.shape, x_dev.shape, x_test.shape)
#x_test = np.array([np.concatenate((i[0:6], i[-6::]),axis = 0)  for i in x_test])

tr_labels = datamain[3]
dev_labels = datamain[4]
test_labels = datamain[5]

f = open('QA_data-1.p', 'rb')
data_aux = pickle.load(f)
f.close()

AUX_train = data_aux[0] #np.concatenate((data_aux[0], data_aux[0],data_aux[0], data_aux[0],data_aux[0], data_aux[0],data_aux[0], data_aux[0],data_aux[0], data_aux[0],data_aux[0], data_aux[0]), axis = 0)
print(AUX_train.shape)
#AUX_train = [np.concatenate((i[0:6], i[-6::]),axis = 0) for i in AUX_train]

AUX_dev = data_aux[1]#np.concatenate((data_aux[1], data_aux[1],data_aux[1], data_aux[1],data_aux[1], data_aux[1],data_aux[1], data_aux[1],data_aux[1], data_aux[1],data_aux[1], data_aux[1]), axis = 0)
print(AUX_dev.shape)
#AUX_dev = [np.concatenate((i[0:6], i[-6::]),axis = 0) for i in AUX_dev]

AUX_test  = data_aux[2] #np.concatenate((data_aux[2],data_aux[2], data_aux[2],data_aux[2], data_aux[2],data_aux[2], data_aux[2],data_aux[2], data_aux[2],data_aux[2], data_aux[2],data_aux[2], data_aux[2]), axis = 0)
#AUX_test = [np.concatenate((i[0:6], i[-6::]),axis = 0) for i in AUX_test]
print(AUX_test.shape)
tr_labelsAUX = data_aux[3] #data_aux[3] + data_aux[3] + data_aux[3] + data_aux[3] +data_aux[3] + data_aux[3] + data_aux[3] + data_aux[3]+data_aux[3] + data_aux[3] + data_aux[3] + data_aux[3]
dev_labelsAUX = data_aux[4] #data_aux[4] +data_aux[4] +data_aux[4] +data_aux[4]+data_aux[4] +data_aux[4] +data_aux[4] +data_aux[4]+data_aux[4] +data_aux[4] +data_aux[4] +data_aux[4]
test_labelsAUX = data_aux[5] #data_aux[5] +data_aux[5] + data_aux[5] + data_aux[5] + data_aux[5]+data_aux[5] + data_aux[5] + data_aux[5] + data_aux[5]+data_aux[5] + data_aux[5] + data_aux[5] + data_aux[5]

print(len(test_labelsAUX))
####STL MODEL#######

input1 = Input(shape=(len(x_train[0]),))
input2 = Input(shape=(len(x_train[0]),))

main_layer1 = Dense(64, activation='relu')(input1)
drop1 = Dropout(0.5)(main_layer1)

aux_layer1 = Dense(64, activation='relu')(input2)
drop2 = Dropout(0.5)(aux_layer1)

# a layer instance is callable on a tensor, and returns a tensor
shared_layer= Dense(64, activation='relu')
shared_out1 = shared_layer(drop1)
shared_out2 = shared_layer(drop2)

drop3 = Dropout(0.5)(shared_out1)
drop4 = Dropout(0.5)(shared_out2)




main_layer2 = Dense(64, activation='relu')(drop3)
drop5 = Dropout(0.5)(main_layer2)



aux_layer2 = Dense(64, activation='relu')(drop4)
drop6 = Dropout(0.5)(aux_layer2)

main_out = Dense(1, activation='sigmoid', name="main")(drop5)
aux_out = Dense(1, activation='sigmoid')(drop6)

batch_size = 100
nb_epoch = 5000

model = Model(inputs=[input1, input2], outputs=[main_out, aux_out])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss=['binary_crossentropy','binary_crossentropy'],
              metrics=['accuracy'],
             loss_weights = [.9, .1])


print(model.summary())
hist = History()
checkpoint = ModelCheckpoint(filepath, monitor='val_main_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, hist]
history = model.fit([np.array(x_train), np.array(AUX_train)[0:len(x_train)]], [np.array(tr_labels), np.array(tr_labelsAUX[0:len(x_train)])],
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,verbose=1,
                    validation_data= ([np.array(x_dev), np.array(AUX_dev[0:len(x_dev)])], [np.array(dev_labels), np.array(dev_labelsAUX[0:len(x_dev)])]),
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
score = best_model.evaluate([np.array(x_test), np.array(AUX_test[0:len(x_test)])],  [np.array(test_labels), np.array(test_labelsAUX[0:len(x_test)])], verbose=0)

print('Test accuracy on main task:', score[-2])

print('Test accuracy on aux task:', score[-1])


probs = best_model.predict([x_test,  np.array(AUX_test[0:len(x_test)])])
#print(probs)
probs_list = [i[0] for i in probs[0]]

predictions = predict_classes(np.array(probs))
preds_list = [i[0] for i in predictions[0]]
print(preds_list)

prob_file = open('PREDS/probs-mtl-mlp-17-NLI', 'wb')
pickle.dump(probs_list, prob_file)
prob_file.close()



pred_file = open('PREDS/preds-mtl-mlp-17-NLI', 'wb')
pickle.dump(preds_list, pred_file)
pred_file.close()


trues = open('PREDS/true-values', 'wb')
pickle.dump(test_labels, trues)
pred_file.close()

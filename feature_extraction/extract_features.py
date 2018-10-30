import argparse
import features as ft
import pandas as pd
import numpy as np



parser = argparse.ArgumentParser(description='Specify paths')
parser.add_argument('glove_path',  type=str,
                     help='Provide the full path to the glove embeddings file')
parser.add_argument('query_path',  type=str,
                     help='Provide the full path to the QQ dataframes')  

arguments = parser.parse_args()                

#load embeddings
print('loading GLOVE embeddings....')
embeddings_path = arguments.glove_path
f = open(embeddings_path)

vocab = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    vocab[word] = coefs

f.close()

print('loading QQ data....')

train = pd.read_csv(arguments.query_path+'/'+'QQ_train.csv', sep = ',') #.dropna(axis=0)
dev = pd.read_csv(arguments.query_path+'/'+'QQ_dev.csv', sep = ',') #.dropna(axis=0)
test = pd.read_csv(arguments.query_path+'/'+'QQ_test.csv', sep = ',') #.dropna(axis=0)

def merge(list1, list2):
    new = []
    for i in range(len(list1)):
        new.append(list1[i]+ " " +list2[i])
    return new


rel_q = merge(train.RELQ_CATEGORY.tolist(),train.RELQ_text.tolist())
org_q = merge(train.ORGQ_SUBJECT.tolist(), train.ORGQ_TEXT.tolist())

#dev

rel_qd = merge(dev.RELQ_CATEGORY.tolist(),dev.RELQ_text.tolist())
org_qd = merge(dev.ORGQ_SUBJECT.tolist(), dev.ORGQ_TEXT.tolist())


#TEST

rel_qt = merge(test.RELQ_CATEGORY.tolist(),test.RELQ_text.tolist())
org_qt = merge(test.ORGQ_SUBJECT.tolist(), test.ORGQ_TEXT.tolist())


#TRAIN


#EMBEDDINGS AVERAGED
rel_qtoken = ft.tokenized(rel_q, vocab)
org_qtoken = ft.tokenized(org_q, vocab)


#dev

rel_qdtoken = ft.tokenized(rel_qd, vocab)
org_qdtoken = ft.tokenized(org_qd, vocab)


#TEST

rel_qttoken = ft.tokenized(rel_qt, vocab)
org_qttoken = ft.tokenized(org_qt, vocab)

#LABELS

tr_labels = ft.get_labels(train)
dev_labels = ft.get_labels(dev)
test_labels = ft.get_labels(test)

## EXTRACT FEATS

#### extract from bow
print('extracting features for similarity features for QQ data')
feats_train = ft.extract_all(rel_q, org_q)
feats_dev = ft.extract_all(rel_qd, org_qd,)
feats_test = ft.extract_all(rel_qt, org_qt)

####extract from embeddings average

feats_train_emb = ft.extract_emb(rel_qtoken, org_qtoken)
feats_dev_emb = ft.extract_emb(rel_qdtoken, org_qdtoken)
feats_test_emb = ft.extract_emb(rel_qttoken, org_qttoken,)

x_train = np.concatenate((feats_train, feats_train_emb), axis = 1)
x_dev = np.concatenate((feats_dev, feats_dev_emb), axis = 1)

x_test = np.concatenate((feats_test, feats_test_emb), axis = 1)


import pickle

print('outputting qq file')
###OUTPUT PICKLE FILE
qq_data = [x_train, x_dev, x_test, tr_labels, dev_labels, test_labels]
f = open(arguments.query_path+'/'+'qq_data.p', 'wb')
pickle.dump( qq_data, f)
f.close()

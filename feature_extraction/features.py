import pandas as pd
import nltk
import keras
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import math




def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def tokenized(list_, vocab):
    print('tokenizing......')
    cleaned_sents = []
    for text in list_:
        cur_text = []
        for sent in sent_tokenize(text.lower()):
            cur_sent = []
            for word in word_tokenize(sent):
                if word in vocab.keys():
                    cur_sent.append(vocab[word])
                else:
                    cur_sent.append(vocab['unk'])
                cur_text.append(cur_sent)
        cleaned_sents.append(np.mean(cur_sent, axis = 0))
    

    return cleaned_sents



def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

##LEXICAL SIMILARITY
def lexical_overlap(rel_q, org_q):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    print('calculating lexical overlap....')

    overlap = []
    for i in range(len(rel_q)):
        rel_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(rel_q[i])]
        org_tokens =  [lemmatizer.lemmatize(word) for word in word_tokenize(org_q[i])]

        min_len = min([len(set(rel_tokens)), len(set(org_tokens))])

        inter = len(intersection(rel_tokens, org_tokens))

        overlap.append(inter/min_len)
    return overlap




def bhattacharyya(a, b):
    
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))

# BAG OF WORDS SIMILARITY

def bow_distance(rel_q, org_q):
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(rel_q + org_q)

    rel_q_vecs = vectorizer.transform(rel_q).toarray()
    org_q_vecs = vectorizer.transform(org_q).toarray()
    import scipy
    from  scipy.spatial.distance import cosine,  euclidean, cityblock

    from scipy.stats import entropy

    print('calculating distances for BOW....')

    cos = []
    euc = []
    man = []
    bhatt = []

    for i in range(len(rel_q_vecs)):
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))

        euc.append(euclidean(rel_q_vecs[i], org_q_vecs[i]))

        man.append(cityblock(rel_q_vecs[i], org_q_vecs[i]))

        bhatt.append(bhattacharyya(rel_q_vecs[i], org_q_vecs[i]))

    return cos, euc, man, bhatt

def boNgrams_distance(rel_q, org_q):
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(3, 3), analyzer='char_wb')
    vectorizer.fit(rel_q + org_q)

    rel_q_vecs = vectorizer.transform(rel_q).toarray()
    org_q_vecs = vectorizer.transform(org_q).toarray()
    import scipy
    from  scipy.spatial.distance import cosine,  euclidean, cityblock
    print('calculating distances for BOW NGRAMS...')
    from scipy.stats import entropy

    cos = []
    euc = []
    man = []
    bhatt = []

    for i in range(len(rel_q_vecs)):
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))
        euc.append(euclidean(rel_q_vecs[i], org_q_vecs[i]))
        man.append(cityblock(rel_q_vecs[i], org_q_vecs[i]))
        bhatt.append(bhattacharyya(rel_q_vecs[i], org_q_vecs[i]))


    return cos, euc, man, bhatt

def get_labels(set_):

   
    bins = []
    for item in set_.RELQ_RELEVANCE2ORGQ.tolist() :
        if item == 'Irrelevant':
            bins.append(0)
        else:
            bins.append(1)

    return bins # keras.utils.to_categorical(bins, num_classes=2)

def get_labelsQA(set_):


    bins = []
    for item in set_.RELC_RELEVANCE2RELQ.tolist() :
        if item == 'Good':
            bins.append(1)
        else:
            bins.append(0)

    return bins # keras.utils.to_categorical(bins, num_classes=2)



def emb_distance(rel_q_vecs, org_q_vecs):
    print('calculating distances for EMBEDDINGS....')

    import scipy
    from  scipy.spatial.distance import cosine,  euclidean, cityblock

    from scipy.stats import entropy

    cos = []
    euc = []
    man = []
    bhatt = []

    for i in range(len(rel_q_vecs)):
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))
        euc.append(euclidean(rel_q_vecs[i], org_q_vecs[i]))
        man.append(cityblock(rel_q_vecs[i], org_q_vecs[i]))
        bhatt.append(bhattacharyya(rel_q_vecs[i], org_q_vecs[i]))


    return cos, euc, man, bhatt

def extract_all(rel_q, org_q, sub_relq, sub_orgq):

    print('extracting all data...')
    cos_t, euc_t , man_t, bhatt_t= bow_distance(rel_q, org_q)
    #cos_s, euc_s , man_s, bhatt_s= bow_distance(sub_relq, sub_orgq)

    cos_t_n3, euc_t_n3, man_t3, bhatt_t3= boNgrams_distance(rel_q, org_q)
    #cos_s_n3, euc_s_n3 , man_s3, bhatt_s3= boNgrams_distance(sub_relq, sub_orgq)

    lex_1 = lexical_overlap(rel_q, org_q)
    lex_2 = lexical_overlap(sub_relq, sub_orgq)

    dataset = pd.DataFrame(data = [cos_t, euc_t , man_t, bhatt_t,  cos_t_n3, euc_t_n3, man_t3, bhatt_t3, lex_1, lex_2])

    return np.array(dataset.transpose())

def extract_emb(rel_q, org_q, sub_relq, sub_orgq):
    cos_t, euc_t , man_t, bhatt_t= emb_distance(rel_q, org_q)
    #cos_s, euc_s , man_s, bhatt_s= emb_distance(sub_relq, sub_orgq)


    dataset = pd.DataFrame(data = [cos_t, euc_t , man_t, bhatt_t])
    return np.array(dataset.transpose())

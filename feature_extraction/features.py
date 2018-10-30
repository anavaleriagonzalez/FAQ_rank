import pandas as pd
import nltk
import keras
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import math
import nltk
from nltk.corpus import stopwords



def words_to_ngrams(words, n, sep=" "):

    string = " ".join(words)
    return [string[i:i+n] for i in range(len(string)-n+1)]

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
    stopwords = nltk.corpus.stopwords.words('english')
    more = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "``", "?", "...", "", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0" ]
    import string
    new_stop = stopwords + more
    from nltk.stem import WordNetLemmatizer
                    
    lemmatizer = WordNetLemmatizer()
    for text in list_:
        #print(text)
        cur_text = []
        for sent in sent_tokenize(text.lower()):
            
            for word in word_tokenize(sent):
                if len(cur_text)== 50:
                    break
                    
                else:
                    dig = word.isdigit()
                    if word in vocab.keys() and word not in new_stop and word not in string.punctuation and dig != True  and len(word.strip("'"))>3:
                    
                        try:
                            int(word)
                            float(word)
                        except Exception:
                            print(word)
                            cur_text.append(vocab[word])
                  
                    
                    
          
        print(len(cur_text))
        if len(cur_text)== 0:
            cur_text.append(vocab['unk'])
       
                    #else:
                    
                
                    #cur_text.append(np.mean(cur_sent, axis = 0))
                    #cur_sent.append(vocab['unk'])
                    
        
       
        c = np.mean(cur_text, axis = 0)
        
    
        cleaned_sents.append(c)
 
    return cleaned_sents



def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

##LEXICAL SIMILARITY
def lexical_overlap(rel_q, org_q):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    print('calculating lexical overlap....')
    stopwords = nltk.corpus.stopwords.words('english')
    more = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "``", "?", "...", ""]
    new = stopwords + more
                
    overlap = []
    for i in range(len(rel_q)):
        rel_tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(rel_q[i]) if word.lower() not in  new]
        org_tokens =  [lemmatizer.lemmatize(word) for word in word_tokenize(org_q[i]) if word.lower() not in  new]

        min_len = min([len(set(rel_tokens)), len(set(org_tokens))])

        inter = len(intersection(rel_tokens, org_tokens))

        overlap.append(inter/min_len)
    return overlap

def lexical_overlap_ngram(rel_q, org_q):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    print('calculating lexical overlap....')
    
    overlap = []
    for i in range(len(rel_q)):
        rel_tokens = words_to_ngrams(word_tokenize(rel_q[i]), 3, sep=" ")
        org_tokens =  words_to_ngrams(word_tokenize(org_q[i]), 3, sep=" ")
        #print(rel_q[i])

        min_len = min([len(set(rel_tokens)), len(set(org_tokens))])

        inter = len(intersection(rel_tokens, org_tokens))
            

        try:
            over = inter/min_len
        except Exception:
            over = 0
        overlap.append(over)
    return overlap



 
def bhattacharyya(a, b):

   
    """ Bhattacharyya distance between distributions (lists of floats). """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    try:
        score = math.log(sum((math.sqrt(abs(u * w)) for u, w in zip(a, b))))
    except Exception:
        score = 0
  
    return score
  
   
    

# BAG OF WORDS SIMILARITY

def bow_distance(rel_q, org_q):
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    stopwords = nltk.corpus.stopwords.words('english')
    more = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
    
    new_stop = stopwords + more
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words=new_stop, ngram_range=(1,3), strip_accents='ascii', analyzer='word', )
    vectorizer.fit(org_q + rel_q)

    rel_q_vecs = vectorizer.transform(rel_q).toarray()
    org_q_vecs = vectorizer.transform(org_q).toarray()
    import scipy
    from  scipy.spatial.distance import cosine,  euclidean, cityblock

    from scipy.stats import entropy
    import sklearn
    

    print('calculating distances for BOW....')

    cos = []
    euc = []
    man = []
    bhatt = []
    kl = []

    for i in range(len(rel_q_vecs)):
       
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))

        euc.append(1-euclidean(rel_q_vecs[i], org_q_vecs[i]))

        man.append(1-cityblock(rel_q_vecs[i], org_q_vecs[i]))

        bhatt.append(bhattacharyya(rel_q_vecs[i], org_q_vecs[i]))
        kl.append(sklearn.metrics.mutual_info_score(rel_q_vecs[i], org_q_vecs[i]))

    return cos, euc, man, bhatt, kl

def boNgrams_distance(rel_q, org_q):
    from gensim import corpora, models, similarities
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import sklearn
                  

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(3, 3), analyzer='char_wb',strip_accents='ascii')
    vectorizer.fit( org_q + rel_q)

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
    kl = []

    for i in range(len(rel_q_vecs)):
       
        
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))
        
        euc.append(1-euclidean(rel_q_vecs[i], org_q_vecs[i]))
        man.append(1-cityblock(rel_q_vecs[i], org_q_vecs[i]))
        bhatt.append(bhattacharyya(rel_q_vecs[i], org_q_vecs[i]))
        kl.append(sklearn.metrics.mutual_info_score(rel_q_vecs[i], org_q_vecs[i]))



    return cos, euc, man, bhatt, kl

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

import numpy as np

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def emb_distance(rel_q_vecs, org_q_vecs):
    print('calculating distances for EMBEDDINGS....')

    import scipy
    from  scipy.spatial.distance import cosine,  euclidean, cityblock

    from scipy.stats import entropy
    import sklearn
    cos = []
    euc = []
    man = []
    bhatt = []
    kl = []
    
   
    for i in range(len(rel_q_vecs)):
      
        cos.append(1- cosine(rel_q_vecs[i], org_q_vecs[i]))
        euc.append(1- euclidean(rel_q_vecs[i], org_q_vecs[i]))
        man.append(1- cityblock(rel_q_vecs[i], org_q_vecs[i]))
        bhatt.append(bhattacharyya(list(rel_q_vecs[i]), list(org_q_vecs[i])))
        kl.append(sklearn.metrics.normalized_mutual_info_score(rel_q_vecs[i], org_q_vecs[i]))


    return cos, euc, man, bhatt, kl

def extract_all(rel_q, org_q):

    print('extracting all data...')
    cos_t, euc_t , man_t, bhatt_t, kl1= bow_distance(rel_q, org_q)
    #cos_s, euc_s , man_s, bhatt_s= bow_distance(sub_relq, sub_orgq)

    cos_t_n3, euc_t_n3, man_t3, bhatt_t3, kl2= boNgrams_distance(rel_q, org_q)
    #cos_s_n3, euc_s_n3 , man_s3, bhatt_s3= boNgrams_distance(sub_relq, sub_orgq)

    lex_1 = lexical_overlap(rel_q, org_q)
    lex_2 = lexical_overlap_ngram(rel_q, org_q)
    
   
     

    dataset = pd.DataFrame(data = [cos_t, euc_t , man_t, bhatt_t,  cos_t_n3, euc_t_n3, man_t3, bhatt_t3, lex_1, lex_2, kl1, kl2])

    return np.array(dataset.transpose())

def extract_emb(rel_q, org_q):
    cos_t, euc_t , man_t, bhatt_t, kl= emb_distance(rel_q, org_q)
    #cos_s, euc_s , man_s, bhatt_s= emb_distance(sub_relq, sub_orgq)


    dataset = pd.DataFrame(data = [cos_t, euc_t , man_t, bhatt_t ])
    print(np.array(dataset.transpose()).shape)
   
    return np.array(dataset.transpose())#np.concatenate((np.array(dataset.transpose()), np.array(rel_q), np.array(org_q)), axis = 1)

# -*- coding: utf-8 -*-
__author__ = "Isabelle Augenstein"
__date__ = "19 November 2017"
__email__ = "augenstein@di.ku.dk"

from gensim.models import word2vec, Phrases
from data_reading import read_chats_xml, read_only_conv, read_FAQs
import logging

def learnMultiword(data, outpath="./models/phrases.model", min_count=5):
    """
    Trains a model that recognises 2-word multi-word expressions, i.e. phrases. If a phrase is recognised, the model
    joins the 2 words with a "_"
    :param data: All the data the model should be trained on. Expects a list of lists, i.e. sentences split into tokens
    :param outpath: path the model should be printed to
    :param min_word_count: Minimum number of words to consider for the Phrases model. For tiny datasets (like the example file) use 2, otherwise 5
    :return: returns the trained model
    """

    print("Learning multiword expressions")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    bigram = Phrases(data, min_count=min_count)
    bigram.save(outpath)

    print("Checking if multiword expressions are detected...")
    test = "jeg kan ikke logge på <unk>"
    sent = test.split(" ")
    print(bigram[sent])

    return bigram


def trainWord2VecModel(data, outpath="./models/word2vec.model", min_word_count=5, num_feats=100):
    """
    Trains a word2vec model that can map words to vectors, which can be used to compute the similarity between words.
    :param data: All the data the model should be trained on. Expects a list of lists, i.e. sentences split into tokens
    :param outpath: path the model should be printed to
    :param min_word_count: Minimum number of words to consider for the word2vec model. For tiny datasets (like the example file) use 2, otherwise 5
    :param num_feats: Dimensionality of the vectors. For large datasets of 100k conversations or more set to 300.
    :return: returns the trained model
    """
    print("Starting word2vec training")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # set parameters - those should work well as default parameters
    num_features = num_feats    # Word vector dimensionality
    min_word_count = min_word_count   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    trainalgo = 1        # cbow: 0 / skip-gram: 1

    print("Training model...")
    model = word2vec.Word2Vec(data, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = trainalgo)

    # add for memory efficiency
    model.init_sims(replace=True)

    # save the model
    model.save(outpath)

    return model


def applyWord2VecMostSimilar(model, word, top=20):
    """
    Demo method - finds the most similar n words for a given word
    :param model: The loaded word2vec model
    :param word: the word
    :param top: how many terms to return
    :return:
    """
    print("Find ", top, " terms most similar to ", word, "...")
    for res in model.wv.most_similar(word, topn=top):
        print(res)
    print("\n")


def applyWord2VecSimilarityBetweenWords(model, w1, w2):
    """
    Determine similarity between words
    :param model: The loaded word2vec model
    :param w1: The first word
    :param w2: The second word
    :return:
    """
    print("Computing similarity between ", w1, " and ", w2, "...")
    print(model.wv.similarity(w1, w2), "\n")
    print("\n")


def applyWord2VecFindWord(model, searchterm):
    """
    Search which words/phrases the model knows which contain a searchterm
    :param model: The loaded word2vec model
    :param searchterm: The word to search for
    :return:
    """
    print("Finding terms containing ", searchterm, "...")
    for v in model.wv.vocab:
        if searchterm in v:
            print(v)
    print("\n")


if __name__ == '__main__':
    # read the example data
    # TODO: remove extra characters when reading data - depending on the data files
    #data = read_chats_xml("./data")
    datafolder = "/home/anavgg/Documents/QA/data/TEST_INPUT/SemEval2016_task3_test_input/English"


    queries, rel_qs, rel_as = read_FAQs(datafolder)

    # train phrases model
    phrase_model = learnMultiword(rel_qs, "./models/phrases.model", 2)

    # add some extra tokens for unknown words
    rel_qs.append(["<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>"])

    # train word2vec model
    word2vecmodel = trainWord2VecModel(phrase_model[rel_qs], "./models/word2vec.model", 2, 100)

    # some demo functions
    applyWord2VecFindWord(word2vecmodel, "bank")
    applyWord2VecMostSimilar(word2vecmodel, "money")
    applyWord2VecSimilarityBetweenWords(word2vecmodel, "bank", "money")


#need to figure out how to match from query to faq data
#1. match to other questions
#2.match to other ANSWERS
    #3. match to answers of relevant questions
    

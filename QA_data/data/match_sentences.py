# -*- coding: utf-8 -*-
__author__ = "Isabelle Augenstein"
__date__ = "19 November 2017"
__email__ = "augenstein@di.ku.dk"

from gensim.models import word2vec, Phrases
from data_reading import read_chats_xml, tokenize, read_only_conv, split_conv, read_FAQs
import numpy as np

def word_in_vocab(word2vec_model, word):
    """
    Check if the word is in the word2vec vocabulary
    :param word2vec_model: the loaded word2vec model
    :param word: the word to be checked
    :return: returns either the word or the placeholder token <unk>
    """
    if word in word2vec_model.wv.vocab:
        return word
    else:
        return "<unk>"


def sentence_matching_word2vec(word2vec_model, source_sentence, target_sentences):
    """
    Simple matching function based on word2vec. For each of the words in the source sentence, get the similarity of
    the word in the target sequence that is most similar to it. The similarity of a sentence is the mean average of the
    similarities of the words.

    :param word2vec_model: loaded word2vec model
    :param source_sentence: source sentences - the sentence to test. The sentence should already be tokenised, i.e. split into its words.
    :param target_sentences: target sentences - the target sentences to test. The target sentences should alreayd be tokenised, i.e. split into its words.
    :return: Similarity map consisting of <ID, similarity> tuples
    """

    sim_map = {}  # ID, similarity

    print("Computing similarities")
    for i, targ_sent in enumerate(target_sentences):
        # for each of the words in the source sentence, get the similarity of the word in the target sequence that is
        # most similar to it
        similarities = [max([word2vec_model.similarity(word_in_vocab(word2vec_model, tok_s), word_in_vocab(word2vec_model, tok_t)) for tok_t in targ_sent])
                            for tok_s in source_sentence]
        sent_sim = np.average(similarities)
        # store in hashmap
        sim_map[i] = sent_sim
    return sim_map


def print_sim_sentences(sim_map, source_sentence, target_sentences, n=5):
    """
    Given the similarity map for target sentences (ID, similarity), prints the most similar n target sentences
    :param sim_map: ID, similarity for each target sentence
    :param source_sentence: Source sentence, tokenised
    :param target_sentences: Target sentences, tokenised
    :param n: Number of sentences to print
    :return: Returns the most similar n sentences to the source sentence in descending order of similarity
    """
    print("Most similar", n, "sentences to source sentence:", " ".join(source_sentence))
    i = 0
    for key, value in sorted(sim_map.items(), key=lambda x:x[1], reverse=True):
        print(str(value) + "\t" + " ".join(target_sentences[key]))
        if i == n:
            break
        i += 1


if __name__ == '__main__':

    # read in the data
    #data = read_chats_xml("./data")
    datafolder = "/home/anavgg/Documents/QA/data/TEST_INPUT/SemEval2016_task3_test_input/English"

    queries, rel_qs, rel_as = read_FAQs(datafolder)

    # load the trained word2vec model
    word2vec_model = word2vec.Word2Vec.load("./models/word2vec.model")

    # match a source sentence to target sentences
    # for the target sentences, we just take all sentences here
    # TODO: it would make sense to only match queries to other queries as opposed to to all sentences in the data
    source_sentence = tokenize("jeg har et problem")
    sim_map = sentence_matching_word2vec(word2vec_model, source_sentence, rel_as)
    print_sim_sentences(sim_map, source_sentence, rel_as)



    #####################################################

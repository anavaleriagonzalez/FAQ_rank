from gensim.models import word2vec, Phrases
import similarity_model
import data_reading
import match_sentences

#first we import data
datafolder = "/home/anavgg/Documents/QA/data/TEST_INPUT/SemEval2016_task3_test_input/English"
queries, rel_qs, rel_as = data_reading.read_FAQs(datafolder)

#printing some demo stuff

print('sample queries')
print(queries[0:5])
print('sample relevant questions')
print(rel_qs[0:5])
print('sample relevant answers')
print(rel_as[0:5])

#training a model on the relevant QUESTIONS
phrase_model = similarity_model.learnMultiword(rel_qs, "./models/phrases_qs.model", 2)

# add some extra tokens for unknown words
rel_qs.append(["<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>"])

# train word2vec model
word2vecmodel = similarity_model.trainWord2VecModel(phrase_model[rel_qs], "./models/word2vec1.model", 2, 100)

# some demo functions
similarity_model.applyWord2VecFindWord(word2vecmodel, "log")
similarity_model.applyWord2VecMostSimilar(word2vecmodel, "learn")
similarity_model.applyWord2VecSimilarityBetweenWords(word2vecmodel, "bank", "money")

#################################################################################
#################################################################################


#training a model on the relevant ANSWERS
phrase_model2 = similarity_model.learnMultiword(rel_as, "./models/phrases_as.model", 2)

# add some extra tokens for unknown words
rel_as.append(["<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>", "<unk>"])

# train word2vec model
word2vecmodel2 = similarity_model.trainWord2VecModel(phrase_model2[rel_as], "./models/word2vec2.model", 2, 100)

# some demo functions
similarity_model.applyWord2VecFindWord(word2vecmodel2, "log")
similarity_model.applyWord2VecMostSimilar(word2vecmodel2, "learn")
similarity_model.applyWord2VecSimilarityBetweenWords(word2vecmodel2, "bank", "money")



##################################################################################
##################################################################################

#source_sentence = data_reading.tokenize("jeg har et problem")
source_sentence = queries[0]
print(queries[0])
sim_map = match_sentences.sentence_matching_word2vec(word2vecmodel,  source_sentence, rel_qs)
match_sentences.print_sim_sentences(sim_map, source_sentence, rel_qs)



#####################################################

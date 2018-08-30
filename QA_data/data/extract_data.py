import os
import re
import xml.etree.ElementTree as ET
import match_sentences
import similarity_model
from data_reading import tokenize

#specification of data folders

datafolder_test = "/home/anavgg/Desktop/QA/data/TEST_INPUT/SemEval2016_task3_test_input/English"

filepaths_test = [os.path.join(datafolder_test, f) for f in os.listdir(datafolder_test) if "xml" in f]
filepaths_test
#############################

datafolder_train = "/home/anavgg/Desktop/QA/data/semeval2016-task3-cqa-ql-traindev-v3.2 ENG/v3.2/train"

filepaths_train = [os.path.join(datafolder_train, f) for f in os.listdir(datafolder_train) if "xml" in f]
filepaths_train

################################
datafolder_dev = "/home/anavgg/Desktop/QA/data/semeval2016-task3-cqa-ql-traindev-v3.2 ENG/v3.2/dev"

filepaths_dev = [os.path.join(datafolder_dev, f) for f in os.listdir(datafolder_dev) if "xml" in f]
filepaths_dev


def get_orgq(filepaths_list):
    new_queries = []
    ##extract the original queries, along with subject and id

    for fil in filepaths_list:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('OrgQuestion'):
            if [content.attrib, content.find('OrgQSubject').text , content.find('OrgQBody').text] not in new_queries :
            #print(root.find('./OrgQuestion').attrib['ORGQ_ID'])
                new_queries.append([content.attrib, content.find('OrgQSubject').text , content.find('OrgQBody').text])

    return new_queries

#gathering all relevant questions
def get_relq(filepaths_list):

    rel_questions = []
    #finding all the relevant questions

    for fil in filepaths_list:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('RelQuestion'):
            if [content.attrib, content.find('RelQSubject').text , content.find('RelQBody').text] not in rel_questions :
                #print(content.attrib)
                rel_questions.append([content.attrib, content.find('RelQSubject').text , content.find('RelQBody').text])
    return rel_questions

train_orgq = get_orgq(filepaths_train)
train_relq = get_relq(filepaths_train)

test_orgq = get_orgq(filepaths_test)
test_relq = get_relq(filepaths_test)


dev_orgq = get_orgq(filepaths_dev)
dev_relq = get_relq(filepaths_dev)

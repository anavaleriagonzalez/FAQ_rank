# -*- coding: utf-8 -*-
import os
import re
import xml.etree.ElementTree as ET

def tokenize(xs, pattern="([\s'\-\.\,\!])"):
    """Splits sentences into tokens by regex over punctuation: ( -.,!])["""
    return [x for x in re.split(pattern, xs)
            if not re.match("\s", x) and x != ""]

def read_chats_xml(datafolder):
    """
    Reads the files in the data folder.
    TODO: Remove non-conversation characters, if appropriate
    :param datafolder: Folder where the data files are located
    :return: The loaded and tokenised data
    """
    data = []
    for f in os.listdir(datafolder):
        with open(os.path.join(datafolder, f), mode='r', encoding='utf-8') as f:
            for l in f:
                # skips empty lines
                if l.strip("\n") == "":
                    continue
                data.append(tokenize(l.strip("\n").lower()))

    return data

def read_only_conv(datafolder):
    '''
    Extracting only the conversations, ignoring the meta data..
    '''
    filepaths= [os.path.join(datafolder, f) for f in os.listdir(datafolder)]

    tokenized_conv = []

    for file1 in filepaths:
        tree = ET.parse(file1)
        root = tree.getroot()

        for content in root.iter('content'):
            if type(content.text) != str:
                pass
            else:
                lines = content.text.split('\n')
                for l in lines:
                    if l.strip("\n") == "":
                        continue
                    tokenized_conv.append(tokenize(l.strip().lower()))

    return tokenized_conv


def read_FAQs(datafolder):

    filepaths = [os.path.join(datafolder, f) for f in os.listdir(datafolder)]
    filepaths
    new_queries = []

    for fil in filepaths:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('OrgQBody'):
            if type(content.text) == str and tokenize(content.text) not in new_queries :
                new_queries.append(tokenize(content.text))
    #gathering all relevant questions
    rel_questions = []
    rel_answers = []

    for fil in filepaths:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('RelQBody'):
            if type(content.text) == str and tokenize(content.text) not in rel_questions :
                rel_questions.append(tokenize(content.text))

    #gathering all relevant answers
    for fil in filepaths:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('RelCText'):
            if type(content.text) == str and tokenize(content.text) not in rel_answers :
                rel_answers.append(tokenize(content.text))

    return new_queries, rel_questions, rel_answers




def split_conv(tokenized_conv):
    ''' not sure if this is useful, but playing around with splitting
    employee sentences and client sentences '''

    client = []
    emp = []
    for i in tokenized_conv:
        if '<b>c1' in i[0]:
            client.append(i)
        else:
            emp.append(i)

    return client, emp


if __name__ == '__main__':
    #data = read_chats_xml("./data")

    #tokenized_convos = read_only_conv('data/')

    #to try later; when a client types in a sentences, find the most similar cleint sentence and
    # suggest the corresponding answer.
    #client, emp = split_conv(tokenized_convos)
    datafolder = "/home/anavgg/Documents/QA/data/TEST_INPUT/SemEval2016_task3_test_input/English"
    queries, rel_qs, rel_as = read_FAQs(datafolder)


    print('sample queries')
    print(queries[0:10])
    print('sample relevant questions')
    print(rel_qs[0:10])

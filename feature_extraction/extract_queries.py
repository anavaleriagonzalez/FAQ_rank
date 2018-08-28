import xml.etree.ElementTree as ET
import pandas as pd


def get_orgq(filepaths_list):
    #extracts the original queries from the file paths
    new_queries = []
    for fil in filepaths_list:
        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('OrgQuestion'):
            if [content.attrib, content.find('OrgQSubject').text , content.find('OrgQBody').text] not in new_queries :
                new_queries.append([content.attrib, content.find('OrgQSubject').text , content.find('OrgQBody').text])

    return new_queries


def get_relq(filepaths_list):
    #extracts the relevant questions from each file
    rel_questions = []
    for fil in filepaths_list:

        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('RelQuestion'):

            if content.find('RelQBody').text:
                if [content.attrib, content.find('RelQBody').text] not in rel_questions :

                    rel_questions.append([content.attrib, content.find('RelQBody').text,  content.find('RelQSubject').text])

            else:
                rel_questions.append([content.attrib, content.find('RelQSubject').text,  content.find('RelQSubject').text])

        print(len(rel_questions))

    return rel_questions

def get_rela(filepaths_list):

    rel_answers = []
    #gathering all relevant answers from each file
    for fil in filepaths_list:

        tree = ET.parse(fil)
        root = tree.getroot()
        for content in root.iter('RelComment'):

            if content.find('RelCText').text:
                if [content.attrib, content.find('RelCText').text] not in rel_answers :
                    #print(content.attrib)
                    rel_answers.append([content.attrib, content.find('RelCText').text])

    #print(rel_answers)
    return rel_answers


def get_all(filepath_dict):
    train_orgq = get_orgq(filepath_dict['train'])
    train_relq = get_relq(filepath_dict['train'])
    train_rela  = get_rela(filepath_dict['train'])

    dev_orgq = get_orgq(filepath_dict['dev'])
    dev_relq  = get_relq(filepath_dict['dev'])
    dev_rela  = get_rela(filepath_dict['dev'])

    test_orgq  = get_orgq(filepath_dict['test'])
    test_relq  = get_relq(filepath_dict['test'])
    test_rela  = get_rela(filepath_dict['test'])

    all_ = {"train": [train_orgq, train_relq, train_rela],
            "test": [test_orgq, test_relq, test_rela],
            "dev": [dev_orgq, dev_relq, dev_rela],
            }

    return all_

def extract_relqdicts(relq):
    only_dicts = {}
    for q in relq:
        q[0]["ORGQ_ID"] = q[0]['RELQ_ID'].split("_")[0]
        q[0]['RELQ_text'] = q[2] + ' ' + q[1]
        q[0]['RelQSubject'] = q[2]
        only_dicts[q[0]["RELQ_ID"]]= q[0]
    #  only_dicts.append(q[0])

    return only_dicts

def extract_reladicts(rela):
    only_dicts = []
    for q in rela:
        q[0]["ORGQ_ID"] = q[0]['RELC_ID'].split("_")[0]
        q[0]['RELC_text'] = q[1]
        q[0]["RELQ_ID"] = q[0]['RELC_ID'].split("_")[0]+'_' + q[0]['RELC_ID'].split("_")[1]

        only_dicts.append(q[0])
    return only_dicts

def extract_orgqdictds(orgq):
    only_org_dicts = {}
    for q in orgq:

        q[0]["SUBJECT"] = q[1]
        q[0]["ORGQ_text"] = q[1] + ' ' + q[2]
        only_org_dicts[q[0]["ORGQ_ID"]]= q[0]

    return only_org_dicts

def org_rel_df(set_rel, set_org):

    only_dicts = extract_relqdicts(set_rel)
    org_rel_df = pd.DataFrame(only_dicts)

    orgs = org_rel_df.ORGQ_ID.tolist()
    only_org_dicts = extract_orgqdictds(set_org)

    new_df = []
    for org in orgs:
        new_df.append([ only_org_dicts[org][ 'SUBJECT'], only_org_dicts[org]['ORGQ_text'] ])

    org_df = pd.DataFrame(new_df, columns= [ "ORGQ_SUBJECT", "ORGQ_TEXT"])
    df = pd.concat([org_rel_df, org_df], axis=1)
    df['genre'] = ['none' for i in range(len(df))]

    df['id_int'] = [i for i in range(len(df))]

    return df[pd.notnull(df['RELQ_RELEVANCE2ORGQ'])]

def org_ans_df(set_ans, set_org):

    only_dicts = extract_reladicts(set_ans)
    org_anse_df = pd.DataFrame(only_dicts)

    orgs = org_anse_df.RELQ_ID.tolist()
    RELC_text = org_anse_df.RELC_text.tolist()
    only_org_dicts = extract_relqdicts(set_org)

    new_df = []
    for i in range(len(orgs)):


        new_df.append([ only_org_dicts[orgs[i]][ 'RelQSubject'], only_org_dicts[orgs[i]]['RELQ_text'] ])

    org_df = pd.DataFrame(new_df, columns= [ "RELQ_SUBJECT", "RELQ_TEXT"])
    df = pd.concat([org_anse_df, org_df], axis=1)
    for i in range(len(df)):
        if df['RELC_RELEVANCE2RELQ'][i] == 'PotentiallyUseful':
            df['RELC_RELEVANCE2RELQ'][i] = 'Bad'

    new = df[df['RELC_RELEVANCE2RELQ'] == 'PotentiallyUseful']
    print(len(new), 'length of PotentiallyUseful')
    return df #df[pd.notnull(df['RELQ_RELEVANCE2ORGQ'])]

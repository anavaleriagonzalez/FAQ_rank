import load_paths
import extract_queries as EQ
import pd_to_text
import pandas as pd
import os

config_file = "config.yml"
datapaths = load_paths.load_configs(config_file)

queries = EQ.get_all(datapaths)  #load dictionary with queries for all sets


#create dataframes for each of the sets for original query to related question
train_df = EQ.org_rel_df(queries['train'][1], queries['train'][0])
test_df = EQ.org_rel_df(queries['test'][1], queries['test'][0])
dev_df = EQ.org_rel_df(queries['dev'][1], queries['dev'][0])

print(train_df.head())
p = os.path.abspath('..')

train_df.to_csv(p+"/QA_data/data_dumps/QQ_train.csv")
test_df.to_csv(p+"/QA_data/data_dumps/QQ_test-17.csv")
dev_df.to_csv(p+"/QA_data/data_dumps/QQ_dev.csv")

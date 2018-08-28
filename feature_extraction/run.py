import load_paths
import extract_queries as EQ
import pd_to_text
import pandas as pd

config_file = "config.yml"
datapaths = load_paths.load_configs(config_file)

queries = EQ.get_all(datapaths)  #load dictionary with queries for all sets


#create dataframes for each of the sets for original query to related question
train_df = EQ.org_rel_df(queries['train'][1], queries['train'][0])

test_df = EQ.org_rel_df(queries['test'][1], queries['test'][0])

dev_df = EQ.org_rel_df(queries['dev'][1], queries['dev'][0])



train_df.to_csv("data_preprocessing/data_dumps/QQ_train.csv")
test_df.to_csv("data_preprocessing/data_dumps/QQ_test-17.csv")
dev_df.to_csv("data_preprocessing/data_dumps/QQ_dev.csv")

pd_to_text.return_text(train_df, "train")
pd_to_text.return_text(test_df, "test")
pd_to_text.return_text(dev_df, "dev")


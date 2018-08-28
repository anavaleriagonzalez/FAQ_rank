import pandas as pd
from nltk.tokenize import sent_tokenize

def return_text(df, set_name):
    
    rel_txt = df.RELQ_text.tolist()
    org_txt = df.ORGQ_TEXT.tolist()

    relevance = df.RELQ_RELEVANCE2ORGQ.tolist()
    rel_rank = df.RELQ_RANKING_ORDER.tolist()

    
    rel_file = "rel_"+set_name + ".txt"
    org_file = "org_"+set_name + ".txt"
    both = 'all_'+set_name+ '.txt'
    bothjson = 'all_'+set_name+ '.json'

    df.to_json('texts/'+bothjson, orient='records', lines=True)

    with open('texts/'+rel_file, "w") as f:
        for text in rel_txt:
           
            f.write(text + "\n")
            #f.write("padding sentence here" + "\n")

    with open('texts/'+org_file, "w") as f:
        for text in org_txt:
            f.write(text + "\n")
            #f.write("padding sentence here" + "\n")

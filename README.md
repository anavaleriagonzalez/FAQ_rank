# FAQ_rank

The code belonging to :

Gonzalez-Garduno, Ana Valeria; Augenstein, Isabelle; Søgaard, Anders. 2018. A strong baseline for question relevancy ranking. Conference on Empirical Methods in Natural Language Processing (EMNLP) 2018. Brussels, Belgium.

the code is being updated...
### Before Running

The code was written in python 3.5 and requires keras (with tensorflow backend). Gensim needs to be installed as well as nltk. The code requires nltk data to be downloaded. If not downloaded already, type the following in the command terminal:

          >> python -c "import nltk; nltk.Download()"
          
### Extracting queries from XML files
The files used to extract the train, dev and test sets are under <b><i> QA_data/semEval_data/ </i></b>...  to extract the queries and relevant information for preprocessing simply go into the directory <b><i>feature_extraction/</b></i> and run :

          >> python run.py
          
the extracted queries and other data will be dumped in <b><i>QA_data/data_dumps/</i></b>




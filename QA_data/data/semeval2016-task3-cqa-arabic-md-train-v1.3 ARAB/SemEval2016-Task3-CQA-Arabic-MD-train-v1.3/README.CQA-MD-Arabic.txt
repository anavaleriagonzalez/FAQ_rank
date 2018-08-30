==============================================================
CQA-MD Arabic corpus for SemEval-2016 Task 3
"Community Question Answering" - "Medical Domain"
Version 1.3: January 20, 2016
==============================================================

This file contains the basic information regarding the CQA-MD Arabic corpus provided for the SemEval-2016 task "Community Question Answering". The current version (1.3, January 20, 2016) corresponds to the release of the Arabic full training + development datasets. All changes and updates on these datasets are reported in Section 1 of this document.


[1] LIST OF VERSIONS

  v1.3 [2016/01/20]: added large unannotated data (163,383 question--answer pairs,
                     including those with annotations)

  v1.2 [2015/12/26]: fixed some minor format issues with the XML files;
                     added DTD at the beginning of the XML files.

  v1.1 [2015/12/08]: added ARABIC DEVELOPMENT data:
                     250 original questions, 7,385 potentially related question/answer (QA) pairs.

  v1.0 [2015/09/30]: distribution of the ARABIC TRAINING data:
                     1,031 original questions, 30,411 potentially related question/answer (QA) pairs.


[2] CONTENTS OF DISTRIBUTION 1.3

We provide the following files:

* README.CQA-MD-Arabic.txt
  this file

* SemEval2016-Task3-CQA-MD-train.xml
  traning data set; 1,031 original questions, 30,411 potentially related question/answer (QA) pairs

* SemEval2016-Task3-CQA-MD-dev.xml
  traning data set; 250 original questions, 7,385 potentially related question/answer (QA) pairs

* UNANNOTATED Arabic data
  * http://alt.qcri.org/semeval2016/task3/data/uploads/Arabic.DataDump.txt.gz
    Unannotated data question--answer Arabic data, comma-separated.


This distribution is directly downloadable from the official SemEval-2016 Task 3 website http://alt.qcri.org/semeval2016/task3/index.php?id=data-and-tools

Licensing: 
- these datasets are free for general research use 
- you should use the following citation in your publications whenever using this resource:

@InProceedings{nakov-EtAl:2016:SemEval,
  author    = {Nakov, Preslav  and  M\`{a}rquez, Llu\'{i}s  and  Magdy, Walid  and  Moschitti, Alessandro  and  Glass, Jim  and  Randeree, Bilal},
  title     = {{SemEval}-2016 Task 3: Community Question Answering},
  booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation},
  series    = {SemEval '16},
  month     = {June},
  year      = {2016},
  address   = {San Diego, California},
  publisher = {Association for Computational Linguistics},
}


[3] SUBTASKS

For ease of explanation, here we give the Arabic subtask of SemEval-2016 Task 3:

Task D: Rerank the correct answers for a new question.
 
Given:
  - a new question (aka the original question), 
  - the set of the first 30 related questions (retrieved by a search engine), each associated with one correct answer (which typically have a size of one or two paragraphs).
 
rerank the 30 question-answer pairs according to their relevance with respect to the original question. We want the "Direct" (D) and the "Relevant" (R) answers to be ranked above "Irrelevant" answers (I); the former two are be considered "Relevant" in terms of evaluation (gold labels are contained in the QArel field of the XML file). We will evaluate the position of "Relevant" answers in the rank, therefore, this is a ranking task.

Unlike the English tasks, here we use 30 answers since the retrieval task is much more difficult, leading to low recall, and the frequency of correct answers is much lower.


[4] DATA FORMAT

The datasets are XML-formatted and the text encoding is UTF-8.

A dataset file is a sequence of examples (questions):

<root>
  <Question> ... </Question>
  <Question> ... </Question>
  ...
  <Question> ... </Question>
</root>

Each <Question> has an ID, e.g., <Question QID="200634">

The internal structure of a <Question> is the following:

<Question ...>
  <Qtext> .... </Qtext>
  <QApair> .... </QApair>
  <QApair> .... </QApair>
  <QApair> .... </QApair>
</Question>

<Qtext> is the text of the question.
<QApair> is a question-answer pair retrieved using a search engine; the task is to judge the relevance of this pair with respect to the question. 

There are about 30 instances of <QApair> per <Question>.


*** QApair ***

<QApair> contains the following attributes:

- ID (QAID): a unique ID of the question-answer pair.

- Relevance (QArel): relevance of the question-answer pair with respect to the <Question>, which is to be predicted at test time:
  
  - "D" (Direct): The question-answer pair contains a direct answer to the original question such that if the user is searching for an answer to the original question <Question>, the proposed question-answer pair would be satisfactory and there will be no need to search any further.
  
  - "R" (Related): The question-answer pair contains an answer to the <Question> that covers some of the aspects raised in the original question, but this is not sufficient to answer it directly. With this question-answer pair, it would be expected that the user will continue the search to find a direct answer or more information.

  - "I" (Irrelevant): The question-answer pair contains an answer that does not relate to the original question <Question>.
  
- Confidence (QAconf): This is the confidence value for the Relevance annotation, based on inter-annotator agreement and other factors. This value is available for the TRAINING dataset only, and it is not available for the DEV and the TEST datasets.


[5] ABOUT THE CQA-MD CORPUS

We generated the CQA-MD Arabic corpus using data from three Arabic medical websites. First, we extracted 1,531 medical questions from http://www.webteb.com/. We then used a number of indexing and retrieval methods to generate a list of potentially relevant question-answer pairs by querying the content of two other websites: http://www.altibbi.com/ and http://consult.islamweb.net/.

For each original question, we retrieved the 30 top-ranked question-answer pairs from our combined index of the other two websites. We then used crowd-sourcing to get annotations about the relevance of each question-answer pair with respect to the corresponding original question. We controlled for quality using hidden tests. We asked for three judgments per example, and we used a combination of majority voting and annotator confidence to select the final label. The average inter-annotator agreement was 81%.

Finally, we divided the data into training, development and testing datasets, based on confidence, where the examples in the test dataset have the highest annotation confidence. We further double-checked and manually corrected some of the annotations for the development and the testing datasets whenever necessary.

Here are some statistics about the datasets:

TRAIN:
- ORIGINAL QUESTIONS:
    - TOTAL:               1,031
- RETRIEVED QUESTION-ANSWER PAIRS:
    - TOTAL:              30,411
    - Direct:                917 (at test time, merged with "Relevant")
    - Relevant:           17,412
    - Irrelevant:         12,082

DEV:
- ORIGINAL QUESTIONS:
    - TOTAL:                 250
- RETRIEVED QUESTION-ANSWER PAIRS:
    - TOTAL:               7,384
    - Direct:                 70 (at test time, merged with "Relevant")
    - Relevant:            1,446
    - Irrelevant:          5,868

  
[6] CREDITS

Task Organizers:

  Preslav Nakov, Qatar Computing Research Institute, HBKU
  Lluís Màrquez, Qatar Computing Research Institute, HBKU
  Alessandro Moschitti, Qatar Computing Research Institute, HBKU
  Walid Magdy, Qatar Computing Research Institute, HBKU
  James Glass, CSAIL-MIT
  Bilal Randeree, Qatar Living

Task website: http://alt.qcri.org/semeval2016/task3/

Contact: semeval-cqa@googlegroups.com

Acknowledgments:
  1. We would like to thank Hamdy Mubarak and Abdelhakim Freihat from QCRI who have contributed a lot to the data preparation.
  2. This research is developed by the Arabic Language Technologies (ALT) group at Qatar Computing Research Institute (QCRI), HBKU, within the Qatar Foundation in collaboration with MIT. It is part of the Interactive sYstems for Answer Search (Iyas) project.

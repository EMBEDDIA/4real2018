# Code for experiments conducted in the paper 'Machine Learning Approach to Bilingual Terminology Alignment: Reimplementation and Adaptation' published in 4REAL workshop at LREC 2018 conference and code for experiments conducted in the paper 'Reimplementation, analysis and adaptation of a term-alignment approach', submitted to the Language Resources and Evaluation Journal #

Please cite the following paper [[bib](http://source.ijs.si/mmartinc/4real2018/blob/master/bibtex.js)] if you use this code:

Andraž Repar, Matej Martinc and Senja Pollak. Machine Learning Approach to Bilingual Terminology Alignment: Reimplementation and Adaptation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). Miyazaki, Japan.


## Installation, documentation ##

Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>
Clone the project from the repository with 'git clone http://source.ijs.si/mmartinc/4real2018.git'<br/>
Install dependencies if needed: pip3 install -r requirements.txt

### First download the 'datasets' folder from [http://source.ijs.si/mmartinc/4real2018](http://source.ijs.si/mmartinc/4real2018) and add it to the project root directory. ###

### To reproduce the results published in both papers, run the code in the command line using following commands: ###



Results for the reproduced approach proposed by Aker et. al:<br/>
python3 main.py --pretrained_dataset aker --filter_trainset False

Results for the only terms that are in GIZA++ approach:<br/>
python3 main.py --pretrained_dataset giza_terms_only --filter_trainset False --giza_only True

Results for the GIZA++ output cleaning approach:<br/>
python3 no_lemmatization.py --pretrained_dataset clean --filter_trainset False

Results for the GIZA++ output cleaning + lemmatization approach:<br/>
python3 main.py --pretrained_dataset clean --filter_trainset False

Results for training set 1:200 approach:<br/>
python3 main.py --pretrained_dataset unbalanced --filter_trainset False

Results for three filtering approaches with different trainset positive/negative ratio:<br/>
python3 main.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1<br/>
python3 main.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 10<br/>
python3 main.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 200

### To reproduce the results for two additional experiments in the paper 'Reimplementation, analysis and adaptation of a term-alignment approach',  run the code in the command line using following commands: ###

Results for the reported term length filtering approach:<br/>
python3 main.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1 --term_length_filter True

Results for the reported Cognate approach:<br/>
python3 main.py --pretrained_dataset cognates --filter_trainset True --trainset_balance 200 --cognates True



### You can also produce your own train and test sets with different pos/neg ratio balances by skipping the --pretrained_dataset argument. Following arguments are available: ###

--trainset_balance : A number argument that defines the ratio between positive and negative examples in trainset, e.g. 200 means that 200 negative examples are generated for every positive example in the train set. Default is 1.<br/>
--testset_balance : A number argument that define the ratio between positive and negative examples in testset, e.g. 200 means that 200 negative examples are generated for every positive term pair in the initial term list.<br/>
                   (except for 600 positive terms that are randomly chosen as positive examples). Default is 200.<br/>
--giza_only : A boolean argument (True or False). Define as True if you only want to use terms that are found in the GIZA++ dictionary. Default is False.<br/>
--filter_trainset: A boolean argument (True or False). Filter positive examples in the train set. Default is False.<br/>
--giza_clean: Use clean version of Giza++ generated dictionary. Default is False.<br/>
--cognates: Improves recall for cognate terms. Default is False.<br/>
--term_length_filter: Additional filter which removes all positively classified terms whose word length do not match.<br/>


### Use the no_lemmatization script if you wish to produce unlemmatized train and test sets. This script also supports Dutch and French as target languages and can be used for reproducing all Dutch and French language experiments published in the paper 'Reimplementation, analysis and adaptation of a term-alignment approach'. Language can be chosen with the 'lang' argument: ###
--lang : Possible values are 'sl', 'fr' and 'nl' for Slovenian, French and Dutch. Default is Slovenian.

Results for the reproduced approach proposed by Aker et. al for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset aker --filter_trainset False --lang fr

Results for the reproduced approach proposed by Aker et. al for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset aker --filter_trainset False --lang nl

Results for the only terms that are in GIZA++ approach for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset giza_terms_only --filter_trainset False --giza_only True --lang fr

Results for the only terms that are in GIZA++ approach for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset giza_terms_only --filter_trainset False --giza_only True --lang nl

Results for training set 1:200 approach for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset False --lang fr

Results for training set 1:200 approach for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset False --lang nl

Results for three filtering approaches with different trainset positive/negative ratio for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1 --lang fr<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 10 --lang fr<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 200 --lang fr

Results for three filtering approaches with different trainset positive/negative ratio for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1 --lang nl<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 10 --lang nl<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 200 --lang nl

Results for the term length filtering approach for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1 --term_length_filter True --lang fr

Results for the term length filtering approach for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset unbalanced --filter_trainset True --trainset_balance 1 --term_length_filter True --lang nl

Results for the  Cognate approach for for English-French alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset cognates --filter_trainset True --trainset_balance 200 --cognates True --lang fr

Results for the  Cognate approach for for English-Dutch alignment:<br/>
python3 no_lemmatization.py --pretrained_dataset cognates --filter_trainset True --trainset_balance 200 --cognates True --lang nl


### You can also use the system for predictions on your own terminology datasets by defining the following arguments: ###
--predict_source : A path to a list of source language terms - one term per line, first line should be "source".<br/>
--predict_target : A path to a list of source language terms - one term per line, first line should be "target".


## Output predictions ##

Output predictions for each of the above configurations are available at:<br/>

http://kt.ijs.si/matej_martinc/4real_results.zip


## Contributors to the code ##

Andraž Repar<br/>
Matej Martinc<br/>
Senja Pollak

* [Knowledge Technologies Department](http://kt.ijs.si), Jožef Stefan Institute, Ljubljana

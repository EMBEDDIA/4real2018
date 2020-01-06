# coding=utf-8

import argparse
import pandas as pd
from nltk import word_tokenize
from collections import defaultdict
import editdistance
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn import pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score
from main import createTrainingAndTestingExamples, longest_common_substring, longest_common_subsequence, isFirstWordTranslated, \
                 isLastWordTranslated, percentageOfTranslatedWords, percentageOfNotTranslatedWords, longestTranslatedUnitInPercentage, \
                 longestNotTranslatedUnitInPercentage, percentageOfCoverage, transcribe, filterTrainSet, build_manual_eval_set, \
                 wordLengthMatch, sourceTermLength, targetTermLength, filterByWordLength
import time
from sklearn import svm


def isWordCognate(s, idx):
    terms = s.split('\t')
    term_source, term_target = terms[0], terms[1]
    word_source, word_target = term_source.split()[idx], term_target.split()[idx]
    word_pair = word_source + '\t' + word_target
    lcs = longest_common_substring(word_pair)
    lgth = max(len(word_source), len(word_target))
    if lgth > 4 and float(len(lcs))/lgth >= 0.8:
        #print(s, lcs, lgth)
        return 1
    return 0


def isWordCovered(x, giza_dict, index):
    terms = x.split('\t')
    term_source, term_target = terms[0], terms[1]
    for word, score in giza_dict[term_source.split()[index]]:
        if word in term_target.split():
            return 1
    lcstr = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    lcsr = float(len(longest_common_subsequence(term_source.split()[index] + '\t' + term_target))) / max(
        len(term_source.split()[index]), len(term_target))
    dice = 2 * float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / (
    len(term_source.split()[index]) + len(term_target))
    nwd = float(len(longest_common_substring(term_source.split()[index] + '\t' + term_target))) / min(
        len(term_source.split()[index]), len(term_target))
    editDistance = 1 - (float(editdistance.eval(term_source.split()[index], term_target)) / max(
        len(term_source.split()[index]), len(term_target)))
    if max(lcstr, lcsr, nwd, dice, editDistance) > 0.7:
        return 1
    return 0


def preprocess(text):
    tokens = word_tokenize(text)
    return " ".join(tokens).lower()


def arrangeData(input):
    dd = defaultdict(list)
    with open(input, encoding='utf8') as f:
        for line in f:
            try:
                source, target, score = line.split()
                source = source.strip('`’“„,‘')
                target = target.strip('`’“„,‘')
                dd[source].append((target, score))
            except:
                pass
                #print(line)

    for k, v in dd.items():
        v = sorted(v, key=lambda tup: float(tup[1]), reverse=True)
        new_v = []
        for word, p in v:
            if (len(k) < 4 and len(word) > 5) or (len(word) < 4 and len(k) > 5):
                continue
            if float(p) < 0.05:
                continue
            new_v.append((word, p))
        dd[k] = new_v
    return dd


def getOnlyGizaDictTerms(term_list, giza_dict, lang):
    src_set = set()
    tar_set = set()
    src_list = []
    tar_list = []
    with open(giza_dict, 'r', encoding="utf8") as f:
        for line in f.readlines():
            try:
                src, tar, _ = line.split()
                src_set.add(src.strip())
                tar_set.add(tar.strip())
            except:
                print(line)

    with open(term_list, 'r', encoding="utf8") as f:
        for i, line in enumerate(f.readlines()[1:]):
            id, src, tar = line.split(';')
            inGiza = True
            for src_word in src.split():
                if src_word.strip() not in src_set:
                    inGiza = False
            for tar_word in tar.split():
                if tar_word.strip() not in tar_set:
                    inGiza = False
            if inGiza:
                src_list.append([src.strip(), i])
                tar_list.append([tar.strip(), i])

    columns = ['Term', 'Freq']
    df_src = pd.DataFrame(src_list, columns=columns)
    df_tar = pd.DataFrame(tar_list, columns=columns)
    df_src.to_csv("term_giza_only_" + lang + "_en.csv", index=False)
    df_tar.to_csv("term_giza_only_" + lang + "_" + lang + ".csv", index=False)



def createFeatures(data, giza_dict, giza_dict_reversed, train=True, cognates=True):
    data['src_term'] = data['src_term'].map(lambda x: preprocess(x))
    data['tar_term'] = data['tar_term'].map(lambda x: preprocess(x))
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    print('preprocessing done')

    data['isFirstWordTranslated'] = data['term_pair'].map(lambda x: isFirstWordTranslated(x, giza_dict))
    data['isLastWordTranslated'] = data['term_pair'].map(lambda x: isLastWordTranslated(x, giza_dict))
    data['percentageOfTranslatedWords'] = data['term_pair'].map(lambda x: percentageOfTranslatedWords(x, giza_dict))
    data['percentageOfNotTranslatedWords'] = data['term_pair'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict))
    data['longestTranslatedUnitInPercentage'] = data['term_pair'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict))
    data['longestNotTranslatedUnitInPercentage'] = data['term_pair'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict))

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordTranslated_reversed'] = data['term_pair'].map(lambda x: isFirstWordTranslated(x, giza_dict_reversed))
    data['isLastWordTranslated_reversed'] = data['term_pair'].map(lambda x: isLastWordTranslated(x, giza_dict_reversed))
    data['percentageOfTranslatedWords_reversed'] = data['term_pair'].map(lambda x: percentageOfTranslatedWords(x, giza_dict_reversed))
    data['percentageOfNotTranslatedWords_reversed'] = data['term_pair'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict_reversed))
    data['longestTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict_reversed))
    data['longestNotTranslatedUnitInPercentage_reversed'] = data['term_pair'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict_reversed))

    data['src_term_tr'] = data['src_term'].map(lambda x: transcribe(x, 'en'))
    data['tar_term_tr'] = data['tar_term'].map(lambda x: transcribe(x, lang))
    data['term_pair_tr'] = data['src_term_tr'] + '\t' + data['tar_term_tr']
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    if cognates:
        data['isFirstWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, 0))
        data['isLastWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, -1))

    data['longestCommonSubstringRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['longestCommonSubsequenceRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_subsequence(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['dice'] = data['term_pair_tr'].map(lambda x: (2 * float(len(longest_common_substring(x)))) / (len(x.split('\t')[0]) + len(x.split('\t')[1])))
    data['NWD'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / min(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['editDistanceNormalized'] = data['term_pair_tr'].map(lambda x: 1 - (float(editdistance.eval(x.split('\t')[0], x.split('\t')[1])) / max(len(x.split('\t')[0]), len(x.split('\t')[1]))))

    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']

    data['isFirstWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, 0))
    data['isLastWordCovered'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict, -1))
    data['percentageOfCoverage'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict))
    data['percentageOfNonCoverage'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict))
    data['diffBetweenCoverageAndNonCoverage'] = data['percentageOfCoverage'] - data['percentageOfNonCoverage']

    if cognates:
        data['wordLengthMatch'] = data['term_pair'].map(lambda x: wordLengthMatch(x))
        data['sourceTermLength'] = data['term_pair'].map(lambda x: sourceTermLength(x))
        data['targetTermLength'] = data['term_pair'].map(lambda x: targetTermLength(x))

    data['term_pair'] = data['tar_term'] + '\t' + data['src_term']

    data['isFirstWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, 0))
    data['isLastWordCovered_reversed'] = data['term_pair'].map(lambda x: isWordCovered(x, giza_dict_reversed, -1))
    data['percentageOfCoverage_reversed'] = data['term_pair'].map(lambda x: percentageOfCoverage(x, giza_dict_reversed))
    data['percentageOfNonCoverage_reversed'] = data['term_pair'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict_reversed))
    data['diffBetweenCoverageAndNonCoverage_reversed'] = data['percentageOfCoverage_reversed'] - data['percentageOfNonCoverage_reversed']
    data['averagePercentageOfTranslatedWords'] = (data['percentageOfTranslatedWords'] + data['percentageOfTranslatedWords_reversed']) / 2

    data = data.drop(['term_pair', 'term_pair_tr', 'src_term_tr', 'tar_term_tr'], axis=1)

    print('feature construction done')
    return data


class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(['src_term', 'tar_term'], axis=1)
        return hd_searches.values


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test a machine translation model')
    parser.add_argument('--pretrained_dataset', type=str, default='', help='Use one of the already generated train and test sets from the data set folder. '
                                                                       'Used for reproduction of results reported in the paper. Argument options are: aker, giza_terms_only, clean and unbalanced')
    parser.add_argument('--trainset_balance', type=str, default='1', help='Define the ratio between positive and negative examples in trainset, e.g. 200 means that 200 negative examples are generated'
                                                                            'for every positive example in the train set')
    parser.add_argument('--testset_balance', type=str, default='200',
                        help='Define the ratio between positive and negative examples in testset, e.g. 200 means that 200 negative examples are generated'
                             'for every positive example in the initial term list.')
    parser.add_argument('--giza_only', default='False', help='Use only terms that are found in the giza corpus')
    parser.add_argument('--filter_trainset', default='False', help='Filter train set')
    parser.add_argument('--lang', default='sl', help='Possible values are sl, fr and nl for Slovene, French and Dutch target language')
    parser.add_argument('--cognates', default='False', help='Approach that improves recall for cognate terms')
    parser.add_argument('--predict_source', default='', help='Use your model on your own dataset. Value should be a path to a list of source language terms')
    parser.add_argument('--predict_target', default='', help='Use your model on your own dataset. Value should be a path to a list of target language terms')
    parser.add_argument('--giza_clean', default='False', help='Use clean version of Giza++ generated dictionary')
    parser.add_argument('--term_length_filter', default='False', help='Additional filter which removes all positively classified terms whose word length do not match')

    start_time = time.clock()

    params = parser.parse_args()
    lang = params.lang

    assert params.pretrained_dataset in ['', 'aker', 'giza_terms_only', 'clean', 'unbalanced', 'cognates'], 'Allowed arguments are: aker, giza_terms_only, clean, unbalanced, cognates'
    assert params.lang in ['sl', 'fr', 'nl'], 'Allowed arguments are: sl, fr, nl'

    pretrained_dataset = params.pretrained_dataset
    trainset_balance = int(params.trainset_balance)
    testset_balance = int(params.testset_balance)
    giza_only = True if params.giza_only == 'True' else False
    giza_clean = True if params.giza_clean == 'True' else False
    filter = True if params.filter_trainset == 'True' else False
    cognates = True if params.cognates == 'True' else False
    term_length_filter = True if params.term_length_filter == 'True' else False
    predict_source = params.predict_source
    predict_target = params.predict_target

    print("Unlemmatized approach, arguments: ")
    print("Pretrained dataset: ", pretrained_dataset)
    print("Trainset balance: ", trainset_balance)
    print("Testset balance: ", testset_balance)
    print("Giza terms only: ", giza_only)
    print("Filter trainset features: ", filter)
    print("Cognates: ", cognates)
    print("Giza clean: ", giza_clean)
    print("Term length filter: ", term_length_filter)
    print("Language: ", lang)


    if len(pretrained_dataset) == 0:
        if lang=="sl":
            dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
            dd_reversed = arrangeData('not_lemmatized/' + lang + '-TransliterationBased.txt')
        else:
            if giza_clean:
                dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
                dd_reversed = arrangeData('not_lemmatized/' + lang + '-enTransliterationBased.txt')
            else:
                dd = arrangeData('not_lemmatized/en-' + lang + 'Unfiltered.txt')
                dd_reversed = arrangeData('not_lemmatized/' + lang + '-enUnfiltered.txt')
        df_terms = pd.read_csv('term_list_' + lang + '.csv', sep=';')
        df_term_freq_tar = set(pd.read_csv('term_giza_only_' + lang + '_' + lang + '.csv')['Term'].tolist())
        df_term_freq_src = set(pd.read_csv('term_giza_only_' + lang + '_en.csv')['Term'].tolist())
        data = createTrainingAndTestingExamples(df_terms, df_term_freq_tar, df_term_freq_src, neg_train_count=trainset_balance, neg_test_count=testset_balance, giza_check=giza_only)
        data_train, data_test = data[0], data[1]
        data_train = createFeatures(data_train, dd, dd_reversed, cognates=cognates)
        data_test = createFeatures(data_test, dd, dd_reversed, train=False, cognates=cognates)

    else:
        if pretrained_dataset == 'aker':
            data_train = pd.read_csv('datasets/train_set_reimplementation_' + lang + '.csv')
            data_test = pd.read_csv('datasets/test_set_reimplementation_' + lang + '.csv')
        elif pretrained_dataset == 'giza_terms_only':
            data_train = pd.read_csv('datasets/train_set_giza_only_' + lang + '.csv')
            data_test = pd.read_csv('datasets/test_set_giza_only_' + lang + '.csv')
        elif pretrained_dataset == 'clean':
            data_train = pd.read_csv('datasets/train_set_giza_clean_' + lang + '.csv')
            data_test = pd.read_csv('datasets/test_set_giza_clean_' + lang + '.csv')
        elif pretrained_dataset == 'unbalanced':
            data_train = pd.read_csv('datasets/train_set_filtering_' + lang + '.csv')
            data_test = pd.read_csv('datasets/test_set_filtering_' + lang + '.csv')
        elif pretrained_dataset == 'cognates':
            data_train = pd.read_csv('datasets/train_set_cognates_' + lang + '.csv')
            data_test = pd.read_csv('datasets/test_set_cognates_' + lang + '.csv')
    if filter:
        data_train = filterTrainSet(data_train, trainset_balance, cognates=cognates)

    print("Train set size: ", data_train.shape)



    # build classification model
    y = data_train['label'].values
    X = data_train.drop(['label'], axis=1)
    if not cognates:
        svm = LinearSVC(C=10, fit_intercept=True)
    else:
        svm = svm.SVC(C=10)


    features = [('cst', digit_col())]

    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
            transformer_list=features,
            n_jobs=1
        )),
        ('scale', Normalizer()),
        ('svm', svm)])

    clf.fit(X, y)

    if not predict_source and not predict_target:

        y = data_test['label'].values
        X = data_test.drop(['label'], axis=1)

        y_pred = clf.predict(X)

        if term_length_filter:
            result = pd.concat([X, pd.DataFrame(y_pred, columns=['prediction']), pd.DataFrame(y, columns=['label'])], axis=1)
            df = filterByWordLength(result)
            y = df['label'].values
            y_pred = df['prediction'].values

        print("Precision on test set:")
        print(precision_score(y, y_pred))

        print("Recall on test set:")
        print(recall_score(y, y_pred))

        print("f1 on test set:")
        print(f1_score(y, y_pred))

        result = pd.concat([X[['src_term', 'tar_term']], pd.DataFrame(y_pred, columns=['prediction'])], axis=1)
        result.to_csv("results/all.csv", encoding="utf8", index=False)
        result = result.loc[result['prediction'] == 1]
        result.to_csv("results/positive.csv", encoding="utf8", index=False)

    else:
        if lang=="sl":
            dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
            dd_reversed = arrangeData('not_lemmatized/' + lang + '-TransliterationBased.txt')
        else:
            dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
            dd_reversed = arrangeData('not_lemmatized/' + lang + '-enTransliterationBased.txt')
            if giza_clean:
                dd = arrangeData('not_lemmatized/en-' + lang + 'TransliterationBased.txt')
                dd_reversed = arrangeData('not_lemmatized/' + lang + '-enTransliterationBased.txt')
            else:
                dd = arrangeData('not_lemmatized/en-' + lang + 'Unfiltered.txt')
                dd_reversed = arrangeData('not_lemmatized/' + lang + '-enUnfiltered.txt')

        l_src = list(pd.read_csv(predict_source, encoding="utf8")['source'].values)
        l_tar = list(pd.read_csv(predict_target, encoding="utf8")['target'].values)

        data_predict = build_manual_eval_set(l_src, l_tar)
        data_predict = createFeatures(data_predict, dd, dd_reversed, train=False, cognates=cognates)
        y_pred = clf.predict(data_predict)

        result = pd.concat([data_predict, pd.DataFrame(y_pred, columns=['prediction'])], axis=1)
        if term_length_filter:
            result = filterByWordLength(result)
        result.to_csv("results/all.csv", encoding="utf8", index=False)
        result = result.loc[result['prediction'] == 1]
        result = result[['src_term', 'tar_term']]
        result.to_csv("results/positive.csv", encoding="utf8", index=False)


    print(time.clock() - start_time, "seconds")



















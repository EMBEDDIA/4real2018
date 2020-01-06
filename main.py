# coding=utf-8

import pandas as pd
from nltk import word_tokenize
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from collections import defaultdict
import editdistance
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn import pipeline
import argparse

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from sklearn import svm


def longest_common_substring(s):
    s = s.split('\t')
    s1, s2 = s[0], s[1]
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]


def longest_common_subsequence(s):
    s = s.split('\t')
    a, b = s[0], s[1]
    lengths = [[0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    # row 0 and column 0 are initialized to 0 already
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x == y:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])
    # read the substring out from the matrix
    result = ""
    x, y = len(a), len(b)
    while x != 0 and y != 0:
        if lengths[x][y] == lengths[x - 1][y]:
            x -= 1
        elif lengths[x][y] == lengths[x][y - 1]:
            y -= 1
        else:
            assert a[x - 1] == b[y - 1]
            result = a[x - 1] + result
            x -= 1
            y -= 1
    return result

def isWordCognate(s, idx):
    terms = s.split('\t')
    term_source_lemma, term_target_lemma = terms[0], terms[1]
    word_source, word_target = term_source_lemma.split()[idx], term_target_lemma.split()[idx]
    word_pair = word_source + '\t' + word_target
    lcs = longest_common_substring(word_pair)
    lgth = max(len(word_source), len(word_target))
    if lgth > 3 and float(len(lcs))/lgth >= 0.7:
        #print(s, lcs, lgth)
        return 1
    return 0


def isFirstWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    firstWordSource = term1.split()[0].strip()
    firstWordTarget = term2.split()[0].strip()
    for target, p in giza_dict[firstWordSource]:
        if target == firstWordTarget:
            return 1

    #fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[firstWordSource]:
            if term2.startswith(target):
                #print(term2, target)
                return 1
    return 0


def isLastWordTranslated(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0], s[1]
    lastWordSource = term1.split()[-1].strip()
    lastWordTarget = term2.split()[-1].strip()
    for target, p in giza_dict[lastWordSource]:
        if target == lastWordTarget:
            return 1

    # fix for compounding problem
    if len(term2) > 4:
        for target, p in giza_dict[lastWordSource]:
            if term2.endswith(target):
                return 1
    return 0



def percentageOfTranslatedWords(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter+=1
                break
    return float(counter)/len(term1)


def percentageOfNotTranslatedWords(s, giza_dict):
    return 1 - percentageOfTranslatedWords(s, giza_dict)


def longestTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter += 1
                if counter > max:
                    max = counter
                break
        else:
            counter = 0
    return float(max) / len(term1)


def longestNotTranslatedUnitInPercentage(s, giza_dict):
    s = s.split('\t')
    term1, term2 = s[0].split(), s[1].split()
    term1 = [x.strip() for x in term1]
    counter = 0
    max = 0
    for word in term1:
        for target, p in giza_dict[word]:
            if target in term2:
                counter = 0
                break
        else:
            counter += 1
            if counter > max:
                max = counter
    return float(max) / len(term1)


def wordLengthMatch(x):
    terms = x.split('\t')
    term_source_lemma, term_target_lemma = terms[0], terms[1]
    if len(term_source_lemma.split()) == len(term_target_lemma.split()):
        return 1
    return 0


def sourceTermLength(x):
    terms = x.split('\t')
    term_source_lemma, _ = terms[0], terms[1]
    return len(term_source_lemma.split())


def targetTermLength(x):
    terms = x.split('\t')
    _, term_target_lemma = terms[0], terms[1]
    return len(term_target_lemma.split())


def isWordCovered(x, giza_dict, index):
    terms = x.split('\t')
    term_source_lemma, term_target_lemma = terms[0], terms[1]
    for word, score in giza_dict[term_source_lemma.split()[index]]:
        if word in term_target_lemma.split():
            return 1
    lcstr = float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / max(len(term_source_lemma.split()[index]), len(term_target_lemma))
    lcsr = float(len(longest_common_subsequence(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / max(len(term_source_lemma.split()[index]), len(term_target_lemma))
    dice = 2 * float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / (len(term_source_lemma.split()[index]) + len(term_target_lemma))
    nwd = float(len(longest_common_substring(term_source_lemma.split()[index] + '\t' + term_target_lemma))) / min(len(term_source_lemma.split()[index]),len(term_target_lemma))
    editDistance = 1 - (float(editdistance.eval(term_source_lemma.split()[index], term_target_lemma)) / max(len(term_source_lemma.split()[index]), len(term_target_lemma)))
    if max(lcstr, lcsr, nwd, dice, editDistance) > 0.7:
        return 1
    return 0


def percentageOfCoverage(x, giza_dict):
    terms = x.split('\t')
    length = len(terms[0].split())
    counter = 0
    for index in range(length):
        counter += isWordCovered(x, giza_dict, index)
    return counter/length


def preprocess(text, lemmatizer, lemmatization=True):
    tokens = word_tokenize(text)
    if lemmatization:
        tokens = [lemmatizer.lemmatize(x) for x in tokens]
    return " ".join(tokens).lower()


def transcribe(text, lang):
    sl_repl = {'č':'ch', 'š':'sh', 'ž': 'zh'}
    en_repl = {'x':'ks', 'y':'j', 'w':'v', 'q':'k'}
    fr_repl = {'é':'e', 'à':'a', 'è':'e', 'ù':'u', 'â':'a', 'ê':'e', 'î':'i', 'ô':'o', 'û':'u', 'ç':'c', 'ë':'e', 'ï':'i', 'ü':'u'}
    nl_repl = {'á':'a', 'é':'e', 'í':'i', 'ó':'o', 'ú':'u', 'ï':'i', 'ü': 'u', 'ë':'e', 'ö':'o', 'à':'a', 'è':'e', 'ĳ':'ij'}
    if lang == 'en':
        en_tr = [en_repl.get(item,item) for item in list(text)]
        return "".join(en_tr).lower()
    elif lang == 'sl':
        sl_tr = [sl_repl.get(item,item)  for item in list(text)]
        return "".join(sl_tr).lower()
    elif lang == 'fr':
        fr_tr = [fr_repl.get(item, item) for item in list(text)]
        return "".join(fr_tr).lower()
    elif lang == 'nl':
        nl_tr = [nl_repl.get(item, item) for item in list(text)]
        return "".join(nl_tr).lower()
    else:
        print ('unknown language for transcription')



def arrangeData(input, lemmatization=False, reverse=False):
    dd = defaultdict(list)
    with open(input, encoding='utf8') as f:
        for line in f:
            line = line.split()
            source, target, score = line[0], line[1], line[2]
            source = source.strip('`’“„,‘')
            target = target.strip('`’“„,‘')
            if lemmatization and not reverse:
                lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
                source = lemmatizer_en.lemmatize(source)
                lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
                target= lemmatizer_sl.lemmatize(target)
            elif lemmatization and reverse:
                lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
                source = lemmatizer_sl.lemmatize(source)
                lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
                target = lemmatizer_en.lemmatize(target)

            dd[source].append((target, score))

    for k, v in dd.items():
        v = sorted(v, key=lambda tup: float(tup[1]), reverse = True)
        new_v = []
        for word, p in v:
            if (len(k) < 4 and len(word) > 5) or (len(word) < 4 and len(k) > 5):
                continue
            if float(p) < 0.05:
                continue
            new_v.append((word, p))
        dd[k] = new_v
    return dd


def createTrainingAndTestingExamples(df_term_pairs, term_freq_tar, term_freq_src, neg_train_count, neg_test_count, giza_check = False):
    term_pairs = list(zip(df_term_pairs['SRC'].tolist(), df_term_pairs['TAR'].tolist()))
    if giza_check:
        term_pairs_giza = []
        for src_term, tar_term in term_pairs:
            # make sure positive examples are in identified terms
            if src_term in term_freq_src and tar_term in term_freq_tar:
                term_pairs_giza.append((src_term, tar_term))
        term_pairs = term_pairs_giza
    tar_terms = [x[1] for x in term_pairs]
    src_terms = [x[0] for x in term_pairs]

    ss = list()
    sm = list()
    mm = list()

    for pair in term_pairs:
        if len(pair[0].split()) == 1 and len(pair[1].split()) == 1:
            ss.append(pair)
        elif len(pair[0].split()) > 1 and len(pair[1].split()) == 1:
            sm.append(pair)
        elif len(pair[0].split()) == 1 and len(pair[1].split()) > 1:
            sm.append(pair)
        elif len(pair[0].split()) > 1 and len(pair[1].split()) > 1:
            mm.append(pair)

    test_set_pairs = []
    random.shuffle(ss)
    random.shuffle(sm)
    random.shuffle(mm)
    for i in range(200):
        test_set_pairs.append(ss.pop())
    for i in range(200):
        test_set_pairs.append(sm.pop())
    for i in range(200):
        test_set_pairs.append(mm.pop())
    term_pairs = ss + sm + mm

    #make sure train and test don't overlap
    existing_pairs = set()

    #build train set
    train_set = []
    for src_term, tar_term in term_pairs:
        counter = neg_train_count
        existing_pairs.add(src_term + '\t' + tar_term)
        train_set.append((src_term, tar_term, 1))
        while counter > 0:
            neg_example = random.choice(tar_terms)
            while (neg_example == tar_term):
                neg_example = random.choice(tar_terms)
            train_set.append((src_term, neg_example, 0))
            existing_pairs.add(src_term + '\t' + neg_example)
            counter -= 1
    df_train = pd.DataFrame(train_set, columns=['src_term', 'tar_term', 'label'])

    #positive examples for test set
    test_set = []
    for src_term, tar_term in test_set_pairs:
        test_set.append((src_term, tar_term, 1))
        existing_pairs.add(src_term + '\t' + tar_term)

    #negative examples for test set
    if neg_test_count < 10:
        pos_eng_terms = random.sample(src_terms, len(test_set) * neg_test_count)
        neg_test_count = 1
    else:
        pos_eng_terms = src_terms
    for src_term in pos_eng_terms:
        counter = neg_test_count
        distinct_neg_terms = set()
        while counter > 0:
            slo_term = random.choice(tar_terms)
            if src_term + '\t' + slo_term not in existing_pairs and slo_term not in distinct_neg_terms:
                test_set.append((src_term, slo_term, 0))
                distinct_neg_terms.add(slo_term)
                counter -= 1
    df_test = pd.DataFrame(test_set, columns=['src_term', 'tar_term', 'label'])
    df_pos = df_test[df_test['label'] == 1]
    df_neg = df_test[df_test['label'] == 0]
    print('number of pos and neg instances in testset: ', df_pos.shape, df_neg.shape)
    return [df_train, df_test]


def filterTrainSet(df, ratio, cognates=False):
    df_pos = df[df['label'] == 1]
    df_pos_dict = df_pos[df_pos['isFirstWordTranslated'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isLastWordTranslated'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isFirstWordTranslated_reversed'] == 1]
    df_pos_dict = df_pos_dict[df_pos['isLastWordTranslated_reversed'] == 1]
    df_pos_dict = df_pos_dict[df_pos['percentageOfCoverage'] > 0.66]
    df_pos_dict = df_pos_dict[df_pos['percentageOfCoverage_reversed'] > 0.66]

    df_pos_dict.reset_index(drop=True, inplace=True)

    if cognates:
        df_pos_cognate_1 = df_pos[df_pos['isFirstWordTranslated'] == 1]
        df_pos_cognate_1 = df_pos_cognate_1[df_pos['isLastWordCognate'] == 1]

        df_pos_cognate_2 = df_pos[df_pos['isLastWordTranslated'] == 1]
        df_pos_cognate_2 = df_pos_cognate_2[df_pos['isFirstWordCognate'] == 1]

        df_pos_cognate_3 = df_pos[df_pos['isFirstWordCognate'] == 1]
        df_pos_cognate_3 = df_pos_cognate_3[df_pos['isLastWordCognate'] == 1]

        df_pos_cognate_1.reset_index(drop=True, inplace=True)
        df_pos_cognate_2.reset_index(drop=True, inplace=True)
        df_pos_cognate_3.reset_index(drop=True, inplace=True)

        df_pos = pd.concat([df_pos_dict, df_pos_cognate_1, df_pos_cognate_2, df_pos_cognate_3])
        df_pos = df_pos.drop_duplicates()
    else:
        df_pos = df_pos_dict


    df_neg = df[df['label'] == 0].sample(frac=1, random_state=123)[:df_pos.shape[0] * ratio]
    df_neg.reset_index(drop=True, inplace=True)

    df = pd.concat([df_pos, df_neg])
    print("Train set shape: ", df.shape)
    return df


def createFeatures(data, giza_dict, giza_dict_reversed, train=True, lemmatization=True, cognates=False):
    lemmatizer_en = Lemmatizer(dictionary=lemmagen.DICTIONARY_ENGLISH)
    data['src_term_lemma'] = data['src_term'].map(lambda x: preprocess(x, lemmatizer_en, lemmatization))
    lemmatizer_sl = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
    data['tar_term_lemma'] = data['tar_term'].map(lambda x: preprocess(x, lemmatizer_sl, lemmatization))
    data['term_pair_lemma'] = data['src_term_lemma'] + '\t' + data['tar_term_lemma']

    print('preprocessing done')

    data['isFirstWordTranslated'] = data['term_pair_lemma'].map(lambda x: isFirstWordTranslated(x, giza_dict))
    data['isLastWordTranslated'] = data['term_pair_lemma'].map(lambda x: isLastWordTranslated(x, giza_dict))
    data['percentageOfTranslatedWords'] = data['term_pair_lemma'].map(lambda x: percentageOfTranslatedWords(x, giza_dict))
    data['percentageOfNotTranslatedWords'] = data['term_pair_lemma'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict))
    data['longestTranslatedUnitInPercentage'] = data['term_pair_lemma'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict))
    data['longestNotTranslatedUnitInPercentage'] = data['term_pair_lemma'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict))

    data['term_pair_lemma'] = data['tar_term_lemma'] + '\t' + data['src_term_lemma']

    data['isFirstWordTranslated_reversed'] = data['term_pair_lemma'].map(lambda x: isFirstWordTranslated(x, giza_dict_reversed))
    data['isLastWordTranslated_reversed'] = data['term_pair_lemma'].map(lambda x: isLastWordTranslated(x, giza_dict_reversed))
    data['percentageOfTranslatedWords_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfTranslatedWords(x, giza_dict_reversed))
    data['percentageOfNotTranslatedWords_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfNotTranslatedWords(x, giza_dict_reversed))
    data['longestTranslatedUnitInPercentage_reversed'] = data['term_pair_lemma'].map(lambda x: longestTranslatedUnitInPercentage(x, giza_dict_reversed))
    data['longestNotTranslatedUnitInPercentage_reversed'] = data['term_pair_lemma'].map(lambda x: longestNotTranslatedUnitInPercentage(x, giza_dict_reversed))
    
    data['src_term_tr'] = data['src_term'].map(lambda x: transcribe(x, 'en'))
    data['tar_term_tr'] = data['tar_term'].map(lambda x: transcribe(x, 'sl'))
    data['term_pair_tr'] = data['src_term_tr'] + '\t' + data['tar_term_tr']
    data['term_pair'] = data['src_term'] + '\t' + data['tar_term']
    #print(data['term_pair_tr'])

    if cognates:
        data['isFirstWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, 0))
        data['isLastWordCognate'] = data['term_pair_tr'].map(lambda x: isWordCognate(x, -1))

    data['longestCommonSubstringRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['longestCommonSubsequenceRatio'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_subsequence(x))) / max(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['dice'] = data['term_pair_tr'].map(lambda x: (2 * float(len(longest_common_substring(x)))) / (len(x.split('\t')[0]) + len(x.split('\t')[1])))
    data['NWD'] = data['term_pair_tr'].map(lambda x: float(len(longest_common_substring(x))) / min(len(x.split('\t')[0]), len(x.split('\t')[1])))
    data['editDistanceNormalized'] = data['term_pair_tr'].map(lambda x: 1 - (float(editdistance.eval(x.split('\t')[0], x.split('\t')[1])) / max(len(x.split('\t')[0]), len(x.split('\t')[1]))))

    data['term_pair_lemma'] = data['src_term_lemma'] + '\t' + data['tar_term_lemma']

    data['isFirstWordCovered'] = data['term_pair_lemma'].map(lambda x: isWordCovered(x, giza_dict, 0))
    data['isLastWordCovered'] = data['term_pair_lemma'].map(lambda x: isWordCovered(x, giza_dict, -1))
    data['percentageOfCoverage'] = data['term_pair_lemma'].map(lambda x: percentageOfCoverage(x, giza_dict))
    data['percentageOfNonCoverage'] = data['term_pair_lemma'].map(lambda x: 1 -percentageOfCoverage(x, giza_dict))
    data['diffBetweenCoverageAndNonCoverage'] = data['percentageOfCoverage'] - data['percentageOfNonCoverage']

    if cognates:
        data['wordLengthMatch'] = data['term_pair_lemma'].map(lambda x: wordLengthMatch(x))
        data['sourceTermLength'] = data['term_pair_lemma'].map(lambda x: sourceTermLength(x))
        data['targetTermLength'] = data['term_pair_lemma'].map(lambda x: targetTermLength(x))

    data['term_pair_lemma'] = data['tar_term_lemma'] + '\t' + data['src_term_lemma']

    data['isFirstWordCovered_reversed'] = data['term_pair_lemma'].map(lambda x: isWordCovered(x, giza_dict_reversed, 0))
    data['isLastWordCovered_reversed'] = data['term_pair_lemma'].map(lambda x: isWordCovered(x, giza_dict_reversed, -1))
    data['percentageOfCoverage_reversed'] = data['term_pair_lemma'].map(lambda x: percentageOfCoverage(x, giza_dict_reversed))
    data['percentageOfNonCoverage_reversed'] = data['term_pair_lemma'].map(lambda x: 1 - percentageOfCoverage(x, giza_dict_reversed))
    data['diffBetweenCoverageAndNonCoverage_reversed'] = data['percentageOfCoverage_reversed'] - data['percentageOfNonCoverage_reversed']

    data['averagePercentageOfTranslatedWords'] = (data['percentageOfTranslatedWords'] + data['percentageOfTranslatedWords_reversed']) / 2

    data = data.drop(['term_pair', 'term_pair_lemma', 'src_term_lemma', 'tar_term_lemma', 'term_pair_tr', 'src_term_tr', 'tar_term_tr'], axis = 1)

    print('feature construction done')
    return data


def build_manual_eval_set(terms_src, terms_tar):
    all_terms = []
    #print(terms_src, terms_tar)
    for src_term in terms_src:
        for tar_term in terms_tar:
            all_terms.append([src_term.strip(), tar_term.strip()])
    df = pd.DataFrame(all_terms)
    #print(df.shape)
    df.columns = ['src_term', 'tar_term']
    return df

def filterByWordLength(df):
    df_pos = df[df['prediction'] == 1]
    df_neg = df[df['prediction'] == 0]
    df_neg.reset_index(drop=True, inplace=True)
    df_pos.reset_index(drop=True, inplace=True)
    for i, row in df_pos.iterrows():
        if len(row['src_term'].split()) != len(row['tar_term'].split()):
            df_pos.set_value(i, 'prediction', 0)
    df = pd.concat([df_pos, df_neg])
    return df


class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(['src_term', 'tar_term'], axis=1)
        return hd_searches.values


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test a machine translation model')
    parser.add_argument('--pretrained_dataset', type=str, default='length',
                        help='Use one of the already generated train and test sets from the data set folder. '
                             'Used for reproduction of results reported in the paper. Argument options are: aker, giza_terms_only, clean and unbalanced')
    parser.add_argument('--trainset_balance', type=str, default='1',
                        help='Define the ratio between positive and negative examples in trainset, e.g. 200 means that 200 negative examples are generated'
                             'for every positive example in the train set')
    parser.add_argument('--testset_balance', type=str, default='200',
                        help='Define the ratio between positive and negative examples in testset, e.g. 200 means that 200 negative examples are generated'
                             'for every positive example in the initial term list.')
    parser.add_argument('--giza_only', default='False', help='Use only terms that are found in the giza corpus')
    parser.add_argument('--giza_clean', default='False', help='Use clean version of Giza++ generated dictionary')
    parser.add_argument('--filter_trainset', default='False', help='Filter train set')
    parser.add_argument('--cognates', default='False', help='Approach that improves recall for cognate terms')
    parser.add_argument('--term_length_filter', default='False', help='Additional filter which removes all positively classified terms whose word length do not match')
    parser.add_argument('--predict_source', default='', help='Use your model on your own dataset. Value should be a path to a list of source language terms')
    parser.add_argument('--predict_target', default='', help='Use your model on your own dataset. Value should be a path to a list of target language terms')

    start_time = time.clock()

    params = parser.parse_args()


    assert params.pretrained_dataset in ['', 'aker', 'giza_terms_only', 'clean', 'unbalanced', 'cognates'], 'Allowed arguments are: aker, giza_terms_only, clean, unbalanced, cognates'

    pretrained_dataset = params.pretrained_dataset
    trainset_balance = int(params.trainset_balance)
    testset_balance = int(params.testset_balance)
    giza_only = True if params.giza_only=='True' else False
    filter = True if params.filter_trainset=='True' else False
    cognates = True if params.cognates == 'True' else False
    giza_clean = True if params.giza_clean == 'True' else False
    predict_source = params.predict_source
    predict_target = params.predict_target
    term_length_filter = True if params.term_length_filter == 'True' else False

    print("Lemmatized approach, arguments: ")
    print("Pretrained dataset: ", pretrained_dataset)
    print("Trainset balance: ", trainset_balance)
    print("Testset balance: ", testset_balance)
    print("Giza terms only: ", giza_only)
    print("Filter trainset features: ", filter)
    print("Cognates: ", cognates)
    print("Term length filter: ", term_length_filter)
    print("Giza clean: ", giza_clean)


    if len(pretrained_dataset) == 0:

        if not giza_clean:
            dd = arrangeData('gdt/gdt.actual.ti.final')
            dd_reversed = arrangeData('gdt/gdt_reverse.actual.ti.final')
        else:
            dd = arrangeData('not_lemmatized/en-slTransliterationBased.txt')
            dd_reversed = arrangeData('not_lemmatized/sl-enTransliterationBased.txt')
        df_terms = pd.read_csv('term_list_sl.csv', sep=';')
        df_term_freq_tar = set(pd.read_csv('term_giza_only_sl_sl.csv')['Term'].tolist())
        df_term_freq_src = set(pd.read_csv('term_giza_only_sl_en.csv')['Term'].tolist())

        data = createTrainingAndTestingExamples(df_terms, df_term_freq_tar, df_term_freq_src, neg_train_count=trainset_balance, neg_test_count=testset_balance, giza_check=giza_only)
        data_train, data_test = data[0], data[1]
        data_train = createFeatures(data_train, dd, dd_reversed, cognates=cognates)
        print(data_test.shape)
        data_test = createFeatures(data_test, dd, dd_reversed, train=False, cognates=cognates)

    else:
        if pretrained_dataset == 'aker':
            data_train = pd.read_csv('datasets/train_set_reimplementation_sl.csv')
            data_test = pd.read_csv('datasets/test_set_reimplementation_sl.csv')
        elif pretrained_dataset == 'giza_terms_only':
            data_train = pd.read_csv('datasets/train_set_giza_only_sl.csv')
            data_test = pd.read_csv('datasets/test_set_giza_only_sl.csv')
        elif pretrained_dataset == 'clean':
            data_train = pd.read_csv('datasets/train_set_giza_clean_lemmatization_sl.csv')
            data_test = pd.read_csv('datasets/test_set_giza_clean_lemmatization_sl.csv')
        elif pretrained_dataset == 'unbalanced':
            data_train = pd.read_csv('datasets/train_set_filtering_sl.csv')
            data_test = pd.read_csv('datasets/test_set_filtering_sl.csv')
        elif pretrained_dataset == 'cognates':
            data_train = pd.read_csv('datasets/train_set_cognates_sl.csv')
            data_test = pd.read_csv('datasets/test_set_cognates_sl.csv')
    if filter:
        data_train = filterTrainSet(data_train, trainset_balance, cognates=cognates)

    print("Train set size: ", data_train.shape[0])

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
        dd = arrangeData('gdt/gdt.actual.ti.final')
        dd_reversed = arrangeData('gdt/gdt_reverse.actual.ti.final')

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





















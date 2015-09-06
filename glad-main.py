# -*- coding: utf-8 -*-

import argparse
import csv
import os
import collections
import math
import random
from collections import Counter
import sys
import zlib

import numpy as np
from nltk import data as nltk_data
from nltk import word_tokenize, FreqDist
from sklearn import svm, preprocessing, naive_bayes, neighbors, ensemble, metrics
from sklearn.cross_validation import StratifiedKFold
from nltk.util import ngrams
from sklearn.externals import joblib
import arff as liac_arff

from logger import log
import evaluation

__authors = ['Benno Weck', 'Manuela Hürlimann', 'Esther Maria van den Berg']
__author__ = ', '.join(__authors)

SENT_DETECTOR = {'EN': nltk_data.load('tokenizers/punkt/english.pickle'),
                 'DU': nltk_data.load('tokenizers/punkt/dutch.pickle'),
                 'GR': nltk_data.load('tokenizers/punkt/greek.pickle'),
                 'SP': nltk_data.load('tokenizers/punkt/spanish.pickle')}
ATTRIBUTES = []  # filled as soon as we know which feature combo is selected
combos = {1: [('5gramSim', 'REAL'), ('4gramSim', 'REAL'), ('3gramSim', 'REAL'), ('2gramSim', 'REAL'),
              ('1gramSim', 'REAL'), ('5gramSpi', 'REAL'), ('4gramSpi', 'REAL'), ('3gramSpi', 'REAL'),
              ('2gramSpi', 'REAL'), ('1gramSpi', 'REAL')] +
             [('PunctSim', 'REAL'), ('LineEndingSim', 'REAL'), ('LineLengthSim', 'REAL'), ('LetterCaseDiff', 'REAL'),
              ('TextBlockDiff', 'REAL')]}
combos[2] = combos[1] + [('CosineSim', 'REAL')]
combos[3] = combos[2] + [('SentLenDiff', 'REAL'), ('jointAvgEnt', 'REAL'), ('EntDiff', 'REAL'),
                         ('CompressionDissimilarity', 'REAL')]
combos[4] = combos[3] + [('kAvgEnt', 'REAL'), ('uAvgEnt', 'REAL'), ('kAvgSentLen', 'REAL'), ('uAvgSentLen', 'REAL')]
combos[5] = [('PunctSim', 'REAL'), ('LineEndingSim', 'REAL'), ('LineLengthSim', 'REAL'), ('LetterCaseDiff', 'REAL'),
             ('TextBlockDiff', 'REAL')]

ARFF_NAME = "training.arff"


def build_dataset(base_path):
    """
    The main method:
    Traverse over the directory tree and read all problem instances, their names, and the corresponding true classes

    :param base_path: The base path to the root directory containing all problem instances
    :return: data instances, problem names, class labels
    """
    truth_dict = collections.defaultdict(str)

    data = []
    problems = []

    for (path, dirs, files) in os.walk(base_path):
        if files:
            if 'truth.txt' in files:
                truth_dict = load_truth_dict(path)

            elif 'unknown.txt' in files:
                language = __get_language_from_dir(os.path.basename(path))

                problem_name = os.path.basename(path)
                problems.append(problem_name)
                log.info("Working on " + problem_name)
                # read only txt files
                known_files = [file for file in files if file.startswith('known') and file.endswith(".txt")]

                instance = read_docs_to_data(path, known_files, 'unknown.txt', language)
                # instance.append(truth_dict[problem_name])
                data.append(instance)

    labels = [truth_dict[p] for p in problems]
    return data, problems, labels


def read_docs_to_data(path, files, file_unknown, language):
    """
    Convert a set of files (a single problem) to an instance
    :param path: The path to the directory containing the text files
    :param files: The file names of the known documents
    :param file_unknown: The filename of the unknown document
    :return: returns a list of feature values
    """
    # token features
    avg_sen_len = avg_ent = avg_joint_ent = vec_sim = compression_dissimilarity = 0
    punct_known = []
    line_endings_known = []
    linelength_known = []
    lettercase_known = []
    textblock_known = []

    if args.average_ngram_sims:
        unigram_sim = bigram_sim = trigram_sim = fourgram_sim = fivegram_sim = unigram_spi = bigram_spi = \
            trigram_spi = fourgram_spi = fivegram_spi = 0
    else:
        unigram_sim = bigram_sim = trigram_sim = fourgram_sim = fivegram_sim = unigram_spi = bigram_spi = \
            trigram_spi = fourgram_spi = fivegram_spi = None

    # 1) extracting data from unknown docs
    document_unknown = open(os.path.join(path, file_unknown), encoding='utf-8-sig').read()
    tokenized_document_unknown = word_tokenize(document_unknown)
    full_training = ""

    # 2) known docs / similarity features
    # token features

    for file in files:
        document = open(os.path.join(path, file), encoding='utf-8-sig').read()
        tokenized_document = word_tokenize(document)
        full_training += document
        avg_sen_len += avg_sent_len(document, len(tokenized_document), language)
        punct_known.append(punct(document))
        line_endings_known.append(line_endings(document))
        linelength_known.append(linelength(document, language))
        lettercase_known.append(lettercase(document))
        textblock_known.append(textblocklen(document))

        avg_ent += entropy(tokenized_document)
        avg_joint_ent += entropy(tokenized_document + tokenized_document_unknown)
        vec_sim += vector_similarity(tokenized_document, tokenized_document_unknown)
        compression_dissimilarity += compression_based_dissimilarity(document, document_unknown)

        if args.average_ngram_sims:
            unigram_sim += char_ngram_similarity(document, document_unknown, 1, 100)
            bigram_sim += char_ngram_similarity(document, document_unknown, 2, 100)
            trigram_sim += char_ngram_similarity(document, document_unknown, 3, 100)
            fourgram_sim += char_ngram_similarity(document, document_unknown, 4, 100)
            fivegram_sim += char_ngram_similarity(document, document_unknown, 5, 500)

            unigram_spi += char_ngram_spi(document, document_unknown, 1, 100)
            bigram_spi += char_ngram_spi(document, document_unknown, 1, 100)
            trigram_spi += char_ngram_spi(document, document_unknown, 1, 100)
            fourgram_spi += char_ngram_spi(document, document_unknown, 1, 100)
            fivegram_spi += char_ngram_spi(document, document_unknown, 1, 100)

    instance = []

    vec_sim /= len(files)
    avg_sen_len /= len(files)
    avg_ent /= len(files)
    avg_joint_ent /= len(files)
    av_punct_known = [sum(i) / len(i) for i in zip(*punct_known)]
    av_line_endings_known = [sum(i) / len(i) for i in zip(*line_endings_known)]
    av_linelength_known = [sum(i) / len(i) for i in zip(*linelength_known)]
    av_lettercase_known = [sum(i) / len(i) for i in zip(*lettercase_known)]
    av_textblock_known = [sum(i) / len(i) for i in zip(*textblock_known)]

    avg_sen_len_unkn = avg_sent_len(document_unknown, len(tokenized_document_unknown), language)
    punct_unknown = punct(document_unknown)
    line_endings_unknown = line_endings(document_unknown)
    linelength_unknown = linelength(document_unknown, language)
    lettercase_unknown = lettercase(document_unknown)
    textblock_unknown = textblocklen(document_unknown)
    compression_dissimilarity /= len(files)

    punct_sim = metrics.pairwise.cosine_similarity(punct_unknown, av_punct_known)  # vectors of length 8
    line_endings_sim = metrics.pairwise.cosine_similarity(line_endings_unknown,
                                                          av_line_endings_known)  # vectors of length 7
    linelength_sim = metrics.pairwise.cosine_similarity(linelength_unknown, av_linelength_known)  # vectors of length 3
    lettercase_diff = np.array(av_lettercase_known) - np.array(lettercase_unknown)  # vectors of length 2
    lettercase_diff = abs(sum(lettercase_diff))
    textblock_diff = np.array(av_textblock_known) - np.array(textblock_unknown)  # vectors of length 2
    textblock_diff = abs(sum(textblock_diff))

    ent_unkn = entropy(tokenized_document_unknown)

    if args.average_ngram_sims:
        unigram_sim /= len(files)
        bigram_sim /= len(files)
        trigram_sim /= len(files)
        fourgram_sim /= len(files)
        fivegram_sim /= len(files)

        unigram_spi /= len(files)
        bigram_spi /= len(files)
        trigram_spi /= len(files)
        fourgram_spi /= len(files)
        fivegram_spi /= len(files)

    else:  # for PAN15 we won't need averaging as 1known:1unknown
        unigram_sim = char_ngram_similarity(full_training, document_unknown, 1, 100)
        bigram_sim = char_ngram_similarity(full_training, document_unknown, 2, 100)
        trigram_sim = char_ngram_similarity(full_training, document_unknown, 3, 100)
        fourgram_sim = char_ngram_similarity(full_training, document_unknown, 4, 100)
        fivegram_sim = char_ngram_similarity(full_training, document_unknown, 5, 500)

        unigram_spi = char_ngram_spi(full_training, document_unknown, 1, 100)
        bigram_spi = char_ngram_spi(full_training, document_unknown, 2, 100)
        trigram_spi = char_ngram_spi(full_training, document_unknown, 3, 100)
        fourgram_spi = char_ngram_spi(full_training, document_unknown, 4, 100)
        fivegram_spi = char_ngram_spi(full_training, document_unknown, 5, 500)

    instance.append(float(fivegram_sim))
    instance.append(float(fourgram_sim))
    instance.append(float(trigram_sim))
    instance.append(float(bigram_sim))
    instance.append(float(unigram_sim))
    instance.append(float(fivegram_spi))
    instance.append(float(fourgram_spi))
    instance.append(float(trigram_spi))
    instance.append(float(bigram_spi))
    instance.append(float(unigram_spi))

    instance.append(float(punct_sim))
    instance.append(float(line_endings_sim))
    instance.append(float(linelength_sim))
    instance.append(float(lettercase_diff))
    instance.append(float(textblock_diff))

    if args.combo >= 2:
        instance.append(float(vec_sim))

    if args.combo >= 3:
        instance.append(float(abs(avg_sen_len - avg_sen_len_unkn)))
        instance.append(float(avg_joint_ent))
        instance.append(float(abs(avg_ent - ent_unkn)))
        instance.append(compression_dissimilarity)

    if args.combo >= 4:
        instance.append(float(avg_ent))
        instance.append(float(ent_unkn))
        instance.append(float(avg_sen_len))
        instance.append(float(avg_sen_len_unkn))

    if args.combo == 5:
        instance = [float(punct_sim), float(line_endings_sim), float(linelength_sim), float(lettercase_diff),
                    float(textblock_diff)]

    return instance


def __get_language_from_dir(dirname):
    if dirname.startswith('EN') or dirname.startswith('EE'):  # English
        return 'EN'
    elif dirname.startswith('DU') or dirname.startswith('DR') or dirname.startswith('DE'):  # Dutch
        return 'DU'
    elif dirname.startswith('SP'):  # Spanish
        return 'SP'
    elif dirname.startswith('GR'):  # Greek
        return 'GR'

    return None


def load_truth_dict(path):
    """
    Load the truth values for a data-set from the TXT file
    :param path: The path to the directory containing the truth.txt file
    :return: A dictionary with the problem names as keys and the true class labels as values
    """

    truth_dict = collections.defaultdict(str)
    with open(os.path.join(path, 'truth.txt'), 'r', encoding='utf-8-sig') as truth_file:
        truth = csv.reader(truth_file, delimiter=' ')
        for problem in truth:
            truth_dict[problem[0]] = problem[1]
            # log.debug(problem[0], problem[1])
        return truth_dict


def compression_based_dissimilarity(doc1, doc2, encoding="utf-8"):
    """
    Get the CDM score (Compression-based Dissimilarity Method) for two documents:
      CDM(x, y) =  (C(x) + C(y)) / C(xy)
      (see Zhensi-Li p. 19)

    Compression algorithm is zlib's gzip.

    :param doc1: the first document (as string)
    :param doc2: the second document (as string)
    :return: the CDM score
    """
    bytes_doc1 = bytes(doc1, encoding)
    bytes_doc2 = bytes(doc2, encoding)
    return (__get_normalised_compressed_size(bytes_doc1) + __get_normalised_compressed_size(bytes_doc2)) / \
           __get_normalised_compressed_size(bytes_doc1 + bytes_doc2)


def __get_normalised_compressed_size(b):
    """
    Calculate normalised compressed size of a byte object 'b'
    :param b:
    :return:
    """
    return len(zlib.compress(b)) / len(b)


def avg_sent_len(document, n_tokens=None, language='EN'):
    """
    Get the average sentences length for a document.
    :param document: The document as string
    :param n_tokens (optional): Number of tokens in document.
    :return: A float
    """
    sents = SENT_DETECTOR[language].tokenize(document)
    if n_tokens is None:
        n_tokens = len(word_tokenize(document))

    return n_tokens / len(sents)


def entropy(tokens):
    """
    Get the Shannon entropy of a document using it's token distribution
    :param tokens: A document represented as a list of tokens.
    :return:
    """
    doc_len = len(tokens)
    frq = FreqDist(tokens)
    for key in frq.keys():
        frq[key] /= doc_len
    ent = 0.0
    for key in frq.keys():
        ent += frq[key] * math.log(frq[key], 2)
    ent = -ent
    return ent


def char_ngram_similarity(doc1, doc2, n, top=100):
    """
    Gives a positive dissimilarity score of two documents with respect to their top m character n-grams distribution.
    If the value is 0 the documents are identical (or at least share an identical top m character n-grams distribution.
    :param doc1:
    :param doc2:
    :param n: the n-gram length
    :param top: Only use the N most frequent n-grams from each document.
    :return: A positive dissimilarity score. If the value is 0 the documents are identical (or at least their top m
             character n-grams distribution.)
    """

    ngrams1 = Counter(ngrams(doc1, n))
    ngrams2 = Counter(ngrams(doc2, n))

    profile1 = [n[0] for n in ngrams1.most_common(top)]
    profile2 = [n[0] for n in ngrams2.most_common(top)]

    # normalise the two ngram distributions
    total1 = np.sum(list(ngrams1.values()))
    for key in ngrams1:
        ngrams1[key] /= total1

    total2 = np.sum(list(ngrams2.values()))
    for key in ngrams2:
        ngrams2[key] /= total2

    # calculate global dissimilarity score
    score = 0
    for n in set(profile1 + profile2):
        f1 = ngrams1[n]
        f2 = ngrams2[n]
        score += ((2 * (f1 - f2)) / (f1 + f2)) ** 2
    return score


def char_ngram_spi(doc1, doc2, n, top=100):
    """
    Gives a "Simple Profile Intersection" (SPI) score of two documents with respect to their top m character n-grams
    distribution.
    SPI is the number of common n-grams among the top m character n-grams.
    :param doc1:
    :param doc2:
    :param n: the n-gram length
    :param top: Only use the N most frequent n-grams from each document.
    :return: The number of common n-grams among the top m character n-grams.
    """

    ngrams1 = Counter(ngrams(doc1, n))
    ngrams2 = Counter(ngrams(doc2, n))

    profile1 = [n[0] for n in ngrams1.most_common(top)]
    profile2 = [n[0] for n in ngrams2.most_common(top)]

    return len(set(profile1).intersection(profile2))


def vector_similarity(doc1_tokens, doc2_tokens):
    """
    Calculate the cosine similarity between two documents.
    The documents are represented as their normalized tf-vectors for the calculation.
    :param doc1_tokens: A list containing the tokens of document 1
    :param doc2_tokens: A list containing the tokens of document 2
    :return: The cosine distance of the two vectors
    """

    vocab = set(doc1_tokens + doc2_tokens)

    frq1 = Counter()
    frq2 = Counter()

    frq1.update(doc1_tokens)
    frq2.update(doc2_tokens)

    v1 = [frq1[term] for term in vocab]
    v2 = [frq2[term] for term in vocab]

    # cosine similarity with included L2 normalisation
    return metrics.pairwise.cosine_similarity(v1, v2)


def punct(document):
    """
    Creates set of counts of punctuation marks
    :param document:
    :return:
    """
    punct_counts = [document.count("!") / len(document), document.count("?") / len(document),
                    document.count(";") / len(document), document.count(":") / len(document),
                    document.count(",") / len(document), document.count(".") / len(document),
                    document.count("-") / len(document), document.count("'") / len(document)]
    return punct_counts


def line_endings(document):
    """
    Counts types of sentence endings
    :param document:
    :return:
    """
    lines = document.split("\n")
    endings = ""
    for i in lines:
        if len(i) > 0:
            endings += i[-1]
    ending_counts = [endings.count(".") / len(lines), endings.count(",") / len(lines), endings.count("?") / len(lines),
                     endings.count(" ") / len(lines), endings.count("!") / len(lines), endings.count("-") / len(lines),
                     endings.count(";") / len(lines)]
    return ending_counts


def linelength(document, language='EN'):
    """
    Count sentences per line, words per line, number of blank lines
    :param document:
    :param language:
    :return:
    """
    sents_per_line = words_per_line = nr_of_blank = 0
    lines = document.split("\n")

    for line in lines:
        if line.isspace():
            nr_of_blank += 0
        else:
            sents_per_line += len(SENT_DETECTOR[language].tokenize(line))

        words_per_line += len(word_tokenize(line))

    words_per_line /= len(lines)
    sents_per_line /= len(lines)
    nr_of_blank /= len(lines)

    return [sents_per_line, words_per_line, nr_of_blank]


def lettercase(document):
    """

    :param document:
    :return:
    """
    capitalletters = len([i for i in document if i.isupper()])

    upper_per_lower = capitalletters / len([i for i in document if i.islower()])
    upper_per_char = capitalletters / len(document)

    lettercase_measures = [upper_per_lower, upper_per_char]

    return lettercase_measures


def textblocklen(document):
    """

    :param document:
    :return:
    """
    lines_per_block = chars_per_block = 0
    blocks = document.split("\n\n")
    for block in blocks:
        lines = block.split("\n")
        lines_per_block += len(lines)
        chars_per_block += len(block)

    lines_per_block /= len(blocks)
    chars_per_block /= len(blocks)

    return [lines_per_block, chars_per_block]


def __scale_features(dataset):
    """
    Check if features should and could be scaled.

    :param dataset: an array of arrays
    :return: scaled feature set if requested/possible, else original dataset.
    """
    if (not args.no_feature_scaling) and clf.scaling_possible:  # scale features if requested, warn if impossible
        return preprocessing.scale(dataset)

    elif (not args.no_feature_scaling) and (not clf.scaling_possible):
        log.warning("Can't scale features with classifier '%s'. Proceeding without feature scaling." % args.clf)
        return dataset


def store_as_arff(data, labels, relation, path, description=u'', attributes=ATTRIBUTES):
    """
    Writes feature data to an ARFF file.
    :param data: The data set. (Without class labels)
    :param labels: Class labels in the same order as the data instances in ``data``
    :param relation: The name of the relation
    :param path: The full path to the file to write to.
    :param description: A description of the relation.
    :param attributes: The attributes in the data set and their levels
    """
    attributes.append(('@@TRUTH@@', ['Y', 'N']))

    new_data = []
    for i in range(len(data)):
        l = list(data[i])
        l += labels[i]
        new_data.append(l)

    arff_data = {
        'description': description,
        'relation': relation,
        'attributes': attributes,
        'data': new_data,
    }
    fh = open(path, 'w')
    fh.write(liac_arff.dumps(arff_data))
    fh.close()


def __available_classifiers():
    available_clfs = dict()
    # features of all available classifiers
    Classifier = collections.namedtuple('Classifier', ['idf', 'full_name', 'function_call',
                                                       'scaling_possible', 'predict_proba', 'numeric_labels'])
    available_clfs["svm"] = Classifier("svm", "Support Vector Machine", svm.SVC(probability=True), True, True, False)
    available_clfs["svm_gs1"] = Classifier("svm", "Co-best SVM according to Skll Grid Search",
                                           svm.SVC(probability=True, kernel="sigmoid", C=0.1, coef0=0.01, gamma=0.01),
                                           True, True, False)
    available_clfs["svm_gs2"] = Classifier("svm", "Co-best SVM according to Skll Grid Search",
                                           svm.SVC(probability=True, kernel="sigmoid", C=0.01, coef0=0.01, gamma=0.0),
                                           True, True, False)
    available_clfs["mnb"] = Classifier("mnb", "Multinomial Naive Bayes", naive_bayes.MultinomialNB(), False, True,
                                       False)  # MNB can't do default scaling: ValueError: Input X must be non-negative
    available_clfs["knn"] = Classifier("knn", "k Nearest Neighbour", neighbors.KNeighborsClassifier(), True, True,
                                       False)  # knn can do feature scaling
    available_clfs["raf"] = Classifier("raf", "Random Forest",
                                       ensemble.RandomForestClassifier(n_estimators=15, max_depth=5, oob_score=True),
                                       True, True, False)
    return available_clfs


def __load_arg_parser():
    arg_parser = argparse.ArgumentParser(description="Perform data manipulation to extract features and finally run "
                                                     "classification.")
    group_get_model = arg_parser.add_mutually_exclusive_group(required=True)
    group_get_model.add_argument("--training", help="The base path to the training set")
    group_get_model.add_argument("-m", "--model", metavar='PATH',
                                 help="Load a pre-trained model and evaluate it on the input data."
                                      "PATH should be a path to a directory containing the model.")
    group_out_dir = arg_parser.add_mutually_exclusive_group()
    group_out_dir.add_argument("-a", "--answers", default="./answers.txt", metavar='PATH',
                               help="Store the answers to a specified file. (Default: ./answers,txt)")
    group_out_dir.add_argument("-o", "--out", metavar="PATH", help="Store the prediction file to a output directory")
    group_test = arg_parser.add_mutually_exclusive_group()
    group_test.add_argument("-i", "--test", "--input", dest="test", help="The base path to the test set")
    group_test.add_argument("--split", nargs='?', type=float, const=0.7,
                            help="Whether to split on the training data. Takes an optional float between 0.0 and 1.0 "
                                 "representing proportion of instances to be used for training.")
    group_test.add_argument("--cv", "--cross-validation", nargs='?', type=int, const=10, metavar='k',
                            help="Perform stratified cross validation on the training data. Takes an optional int "
                                 "specifying the number of folds to perform. (Default k=10)")
    arg_parser.add_argument("-c", "--classifier", dest="clf", default="svm",
                            help=u"The classifier to use. Available: {0:s}. (Default: 'svm')".format(", ".join(
                                available_classifiers.keys())))
    arg_parser.add_argument("--save_model", metavar='PATH',
                            help="Store (pickle) the trained model. PATH should be a path to a file. Will create "
                                 "multiple files in the same directory.")
    arg_parser.add_argument("--no_feature_scaling", action="store_true", help="Don't do feature scaling.")
    arg_parser.add_argument("--no_average_ngram_sims", dest="average_ngram_sims", action="store_false",
                            default=True, help="Use full profile comparison (not average n-gram similarity)")
    arg_parser.add_argument("--combo", type=int, choices=[1, 2, 3, 4, 5], default=4, metavar='i',
                            help="Select a feature combo.")
    return arg_parser


if __name__ == "__main__":
    available_classifiers = __available_classifiers()
    parser = __load_arg_parser()
    args = parser.parse_args()

    # checking arguments
    # training & testing vs splitting vs pre-trained model
    if args.split is not None and (not ((0.0 < args.split) and (args.split <= 1.0))):
        parser.error("--split must be between 0.0 and 1.0")

    if args.split is None and args.test is None and args.save_model is None:
        log.debug("No test data found... Do you maybe want to train and *store* a model?")

    clf = None
    try:  # valid classifier?
        clf = available_classifiers[args.clf.lower()]
    except KeyError:
        parser.error(u"Invalid --classifier argument '{0:s}'. Available classifiers are:"
                     u"{1:s}.".format(args.clf, ", ".join(available_classifiers.keys())))

    ATTRIBUTES = combos[args.combo]
    FEATURE_LABELS = [attr[0] for attr in ATTRIBUTES]
    log.debug(u"Selected Attributes:\n {0:s}.".format(", ".join(FEATURE_LABELS)))

    # preparing data
    if args.split:  # if splitting of 1 data set into train and test
        log.info('Splitting data set %s' % args.split)
        full, full_names, full_labels = build_dataset(args.training)

        # take a random sample of size args.split for training
        n_docs = round(args.split * len(full))
        train_indices = set(random.sample(range(len(full)), n_docs))
        test_indices = set(range(len(full))) - train_indices
        assert len(train_indices) + len(test_indices) == len(full)

        train = [full[i] for i in train_indices]
        test = [full[i] for i in test_indices]

        # retrieve test names based on indices
        test_names = [t for t in full_names if full_names.index(t) not in train_indices]

        # retrieve labels based on indices
        train_labels = []
        test_labels = []
        for idx, label in enumerate(full_labels):
            if idx in train_indices:
                train_labels.append(label)
            else:
                test_labels.append(label)
        log.debug(train_labels)

        train = __scale_features(train)
        test = __scale_features(test)

        log.debug(train_labels)

    elif args.cv:  # if splitting of 1 data set into train and test
        log.info(u'Performing stratified CV with k: {0:d}'.format(args.cv))
        full, full_names, full_labels = build_dataset(args.training)
        test = test_names = test_labels = None

        full = np.array(__scale_features(full))
        full_labels = np.array(full_labels)
        full_names = np.array(full_names)
        classifier = clf.function_call

        answers_path = args.answers
        if args.out:
            answers_path = os.path.join(args.out, "answers.txt")
        if args.out and not os.path.exists(args.out):
            os.makedirs(args.out)

        # remove answers if it already
        if os.path.isfile(answers_path):
            os.remove(answers_path)

        answers = open(answers_path, "a", encoding='utf-8-sig')

        combined_scores = []
        skf = StratifiedKFold(full_labels, args.cv)
        for train, test in skf:
            log.info("%s %s" % (train, test))
            X_train, X_test = full[train], full[test]
            y_train, y_test = full_labels[train], full_labels[test]
            names_train, names_test = full_names[train], full_names[test]

            classifier.fit(X_train, y_train)

            probabilities = classifier.predict_proba(X_test)
            log.debug(probabilities)

            # find which of the two classes is the "Y" / 1 class
            if clf.numeric_labels:
                idx_Y = np.where(classifier.classes_ == 1)
            else:
                idx_Y = np.where(classifier.classes_ == "Y")

            # for each instance, output instance name & probability of "Y" class
            for name, prob in dict(zip(names_test, probabilities)).items():
                prob_Y = prob[idx_Y]
                log.debug(prob_Y)
                answers.write("%s %f\n" % (name, prob_Y))

        log.info(u"Wrote answers to '{0:s}'".format(answers_path))
        answers.close()

        # find truth file and evaluate results
        truth_path = os.path.join(args.training, "truth.txt")
        print(evaluation.main(truth_path, answers_path), file=sys.stderr)

        # and done
        log.info("End after cross validation.")
        sys.exit(0)  # End after CV.

    elif args.training:  # generate train & test instance arrays from train & test
        log.info("Running on {}".format(args.training))
        test = test_names = test_labels = None

        train, _, train_labels = build_dataset(args.training)
        train = __scale_features(train)
        log.debug(train)
        log.info("Length of feature set: {}".format(len(train[0])))
        log.debug('TRAIN LABELS:')
        log.debug(train_labels)

        if args.test:
            log.info("Running on {}".format(args.test))

            test, test_names, test_labels = build_dataset(args.test)
            test = __scale_features(test)
            log.debug('TEST LABELS:')
            log.debug(test_labels)

    else:  # we have a pre-trained model so we only need to process test data
        log.info("Running on {}".format(args.test))
        test, test_names, test_labels = build_dataset(args.test)
        test = __scale_features(test)
        train = train_labels = None

    # change labels to numeric if required
    if clf.numeric_labels:
        log.info("Converting labels to numeric....")
        train_labels = [1 if x == "Y" else 0 for x in train_labels]
        test_labels = [1 if x == "Y"  else 0 for x in test_labels]

    # do classification
    classifier = clf.function_call
    if args.training:
        classifier.fit(train, train_labels)
    if args.model:
        classifier = joblib.load(os.path.join(args.model, "model.pickle"))

    log.info('CLASSIFIER:')
    log.info(classifier)

    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        else:
            log.info(u"Model in {0:s} will get updated ".format(args.save_model))
        joblib.dump(classifier, os.path.join(args.save_model, "model.pickle"))
        info = open(os.path.join(args.save_model, "info.txt"), "w", encoding='utf-8')
        info.write(' '.join(sys.argv))
        info.close()
        log.info(u"Dumped the model to: {0:s}".format(args.save_model))

    # do prediction & write answers file
    if args.split or args.test:
        answers_path = args.answers
        if args.out:
            answers_path = os.path.join(args.out, "answers.txt")
        if args.out and not os.path.exists(args.out):
            os.makedirs(args.out)
        answers = open(answers_path, "w", encoding='utf-8-sig')
        log.info(u"Writing answers to '{0:s}'".format(answers_path))

        # a) classifiers which can do probabilities
        if clf.predict_proba:
            probabilities = classifier.predict_proba(test)
            log.debug(probabilities)

            # find which of the two classes is the "Y" / 1 class
            if clf.numeric_labels:
                idx_Y = np.where(classifier.classes_ == 1)
            else:
                idx_Y = np.where(classifier.classes_ == "Y")

            # for each instance, output instance name & probability of "Y" class
            for name, prob in dict(zip(test_names, probabilities)).items():
                prob_Y = prob[idx_Y]
                log.debug(prob_Y)
                answers.write("%s %f\n" % (name, prob_Y))

        # b) classifiers which cannot predict probabilities
        else:
            # predict
            predicted = classifier.predict(test)
            maxi = max(predicted)
            mini = min(predicted)
            predicted = [(i - ((maxi + mini) / 2)) * (1 / (maxi - mini)) + 0.5 for i in predicted]
            log.debug(predicted)

            # for each instance, output instance name & scaled score
            for name, prob_Y in dict(zip(test_names, predicted)).items():
                answers.write("%s %f\n" % (name, prob_Y))

# -*- coding: utf-8 -*-

import argparse
import csv
import warnings

import sklearn.metrics


def calculateAnswers(truth_dict, answers_dict, misclass_file=None):
    correct = 0
    incorrect = 0
    unanswered = 0

    if misclass_file is not None:
        misclass = open(misclass_file, "w")

    for problem in answers_dict.keys():

        truth = truth_dict[problem]
        answer = answers_dict[problem]

        if truth == "Y" and float(answer) > 0.5:
            correct += 1
        elif truth == "N" and float(answer) < 0.5:
            correct += 1
        elif float(answer) == 0.5:
            unanswered += 1
        else:
            incorrect += 1
            if misclass_file is not None:
                misclass.write(problem + "\n")

    return correct, incorrect, unanswered


def calculateScore(correct, unanswered, numberOfProblems, truth_dict, answers_dict):

    score = (1 / numberOfProblems) * (correct + (unanswered * (correct / numberOfProblems)))

    l_true = []
    l_answers = []

    for problem in answers_dict.keys():

        truth = truth_dict[problem]
        answer = answers_dict[problem]

        if truth == "Y":
            l_true.append(1)
        elif truth == "N":
            l_true.append(0)

        l_answers.append(float(answer))
    
    aucScore = sklearn.metrics.roc_auc_score(l_true, l_answers)

    return round(score, 3), round(aucScore, 3)


def _read_file_to_dict(path):
    """
    Load the problems and the corresponding labels from the *.txt file.
    :param path: The full path to the file to read
    :return: The dictionary with the problem names as keys and the true class labels as values
    """
    label_dict = {}
    with open(path, 'r', encoding='utf-8-sig') as truth_file:
        truth = csv.reader(truth_file, delimiter=' ')
        for problem in truth:
            label_dict[problem[0]] = problem[1]
        return label_dict


def main(truth_file, answers_file, misclass_file=None):
    truth_dict = _read_file_to_dict(truth_file)
    answers_dict = _read_file_to_dict(answers_file)

    if truth_dict.keys() != answers_dict.keys():
        warnings.warn("Apparently there are different problem instances in the truth file and the answers file!")

    if misclass_file is not None:
        correct, incorrect, unanswered = calculateAnswers(truth_dict, answers_dict, misclass_file)
    else:
        correct, incorrect, unanswered = calculateAnswers(truth_dict, answers_dict)

    number_of_problems = correct + incorrect + unanswered

    score, aucScore = calculateScore(correct, unanswered, number_of_problems, truth_dict, answers_dict)
    combined_score = score * aucScore

    print(u"The number of problems: {0:d}".format(number_of_problems))
    print(u"The number of correct answers: {0:d}".format(correct))
    print(u"The number of incorrect answers: {0:d}".format(incorrect))
    print(u"The number of unanswered problems: {0:d}".format(unanswered))
    print(u"The c@1 score: {0:f}".format(score))
    print(u"The AUC: {0:f}".format(aucScore))
    print(u"The combined score of c@1*AUC: {0:f}".format(combined_score))

    return combined_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the PAN scores.")
    parser.add_argument("-a", "--answers", default="./answers.txt", metavar='PATH', dest='answers_file',
                        help="Use a specified file to get the answers. (Default: ./answers,txt)")
    parser.add_argument("-t", "--truth", default="./truth.txt", metavar='PATH', dest='truth_file',
                        help="Use a specified file to get the truth labels. (Default: ./truth,txt)")
    parser.add_argument("-m", "--misclassified", default="./misclassified.txt", metavar='PATH', dest='misclass_file',
                        help="Use a specified file to write misclassified instances. (Default: ./misclassified.txt)")
    args = parser.parse_args()

    main(args.truth_file, args.answers_file, args.misclass_file)

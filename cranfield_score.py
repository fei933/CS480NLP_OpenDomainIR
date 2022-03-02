#!/usr/bin/env python3

import sys
import re

total_documents = 1400


def mean(lst):
    return sum(lst) / len(lst)


def make_key_dict(key_file_name):
    key_dict = dict()
    with open(key_file_name) as key_file:
        for line in key_file.readlines():
            stripped = line.rstrip(" \r\n")
            query, abstract, score = re.split(" +", stripped)
            query = int(query)
            abstract = int(abstract)
            if abstract <= total_documents:
                try:
                    if abstract not in key_dict[query]:
                        key_dict[query].append(abstract)
                except KeyError:
                    key_dict[query] = [abstract]
    return key_dict


def make_response_dict(resp_file_name):
    resp_dict = dict()
    with open(resp_file_name) as resp_file:
        for line in resp_file.readlines():
            stripped = line.rstrip(" \r\n")
            query, abstract, score = re.split(" +", line)
            try:
                query = int(query)
                abstract = int(abstract)
                score = float(score)
            except ValueError as e:
                print('Warning: Each line should consist of 3 numbers with a space in between')
                print('This line does not seem to comply:',line)
                raise e
            try:
                if not abstract in resp_dict[query]:
                    resp_dict[query].append(abstract)
            except KeyError:
                resp_dict[query] = [abstract]
    return resp_dict


def count_correct(keys, responses):
    return sum(1 for doc in responses if doc in keys)


def avg_precision(keys, responses):
    total_answers = len(keys)
    correct = 0
    incorrect = 0
    milestone = .1
    precisions = []

    for abstract in responses:
        if abstract in keys:
            correct += 1
            recall = correct / total_answers
            while recall > milestone:
                precisions.append(correct / (correct + incorrect))
                milestone += .1
        else:
            incorrect += 1
    return mean(precisions)


def precision(keys, responses):
    return count_correct(keys, responses) / len(responses)


def recall(keys, responses):
    return count_correct(keys, responses) / len(keys)


def f_score(prec, rec):
    try:
        return 2 / ((1 / prec) + (1 / rec))
    except ZeroDivisionError:
        return 0


def grade_responses(keys, responses):
    average_precision = avg_precision(keys, responses)

    total_precision = precision(keys, responses)
    total_recall = recall(keys, responses)
    total_f = f_score(total_precision, total_recall)

    trunc_responses = responses[0:len(keys)]
    trunc_precision = precision(keys, trunc_responses)
    trunc_recall = recall(keys, trunc_responses)
    trunc_f = f_score(trunc_precision, trunc_recall)

    return average_precision, total_precision, total_recall, total_f, trunc_precision, trunc_recall, trunc_f


def score (keyFileName, responseFileName):
    key_dict = make_key_dict(keyFileName)
    response_dict = make_response_dict(responseFileName)
    
    all_avg_precisions = []
    all_total_precisions = []
    all_total_recalls = []
    all_total_f = []
    all_trunc_p = []
    all_trunc_r = []
    all_trunc_f = []
    missing_responses = []

    for query in key_dict:
        try:
            avg_p, total_p, total_r, total_f, trunc_p, trunc_r, trunc_f = grade_responses(key_dict[query], response_dict[query])
            all_avg_precisions.append(avg_p)
            all_total_precisions.append(total_p)
            all_total_recalls.append(total_r)
            all_total_f.append(total_f)
            all_trunc_p.append(trunc_p)
            all_trunc_r.append(trunc_r)
            all_trunc_f.append(trunc_f)
        except KeyError:
            missing_responses.append(query)

    if missing_responses:
        print (f"Queries with no responses: {missing_responses}")
    print (f'Mean Average Precision is: {mean(all_avg_precisions)}')
    print(f"Mean precision is {mean(all_total_precisions)}")
    print(f"Mean recall is {mean(all_total_recalls)}")
    print(f"Mean f-score is {mean(all_total_f)}")
    print(f"Mean truncated precision is {mean(all_trunc_p)}")
    print(f"Mean truncated recall is {mean(all_trunc_r)}")
    print(f"Mean truncated f-score is {mean(all_trunc_f)}")


if __name__ == '__main__':
    score(*sys.argv[1:])

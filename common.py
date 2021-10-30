# -*- coding: utf-8 -*-

import heapq
import os
import numpy as np
import random
import datetime


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


def read_file(filename):
    return [line.strip() for line in open(filename).readlines()]


def write_file(filename, content):
    open_file(filename, mode="w").write(content)


def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


# 余弦距离矩阵：不允许负数出现
def cos_distance_metric(vecs):
    cos_dis_metric = np.zeros([len(vecs), len(vecs)])
    for i in range(len(vecs)):
        if i % 1000 == 0:
            print(i, "is processing...")

        cos_dis_metric[i][i] = 1

        for j in range(i + 1, len(vecs)):
            cos_dis = get_cos_similar(vecs[i], vecs[j])
            cos_dis_metric[i][j] = cos_dis
            cos_dis_metric[j][i] = cos_dis
    return cos_dis_metric


def n_largest(arr, n):
    max_number = heapq.nlargest(n, arr)
    max_index = []
    for t in max_number:
        index = arr.index(t)
        max_index.append(index)
        arr[index] = 0
    return dict(zip(max_index, max_number))


def write_res(dir_name, clusters, pure_questions, cnt_threshold):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for i in range(len(clusters)):
        cs = clusters[i]
        if len(cs) < cnt_threshold:
            continue
        file_w = open_file(dir_name + "/" + str(i) + ".txt", mode="w")
        for j in cs:
            file_w.write(pure_questions[j])


def random_sample(lines, fraction=0.1):
    random.shuffle(lines)
    len_test = int(len(lines) * fraction)
    lines_test = lines[0:len_test]
    lines_train = lines[len_test:]
    train_w = open_file("data/train.txt", mode="w")
    test_w = open_file("data/test.txt", mode="w")
    for i in lines_train:
        train_w.write(i)
    for j in lines_test:
        test_w.write(j)


def write_lines(filename, list_res):
    test_w = open_file(filename, mode="w")
    for j in list_res:
        test_w.write(j + "\n")


def time_cost(start):
    return (datetime.datetime.now() - start).microseconds / 1000

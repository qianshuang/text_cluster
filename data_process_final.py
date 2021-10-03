# -*- coding: utf-8 -*-

from common import *
import string
from bert_serving.client import BertClient
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import re


def process_ori_data():
    final_lines = []
    with open_file("data/questions.txt") as f:
        lines = f.readlines()
    print(len(lines))

    for i in range(len(lines)):
        if i % 1000 == 0:
            print(str(i) + " is processing...")
        line = lines[i]

        # 1. 转小写
        lower = line.strip().lower()
        # 2. 去标点
        remove_punc = lower.translate(str.maketrans("", "", string.punctuation))
        # 3. 分词
        tokens = nltk.word_tokenize(remove_punc)
        # 4. 去除停用词
        remove_sw = [w for w in tokens if w not in stopwords.words('english')]

        if len(remove_sw) >= 5 and len(tokens) <= 25:
            final_lines.append(line)

    final_lines = list(set(final_lines))
    print(len(final_lines))
    file_w = open_file("data/pure_questions.txt", mode="w")
    for i in final_lines:
        file_w.write(i)


def sent_to_vec():
    with open_file("data/pure_questions.txt") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    bert_vecs = bc.encode(lines)
    np.save("data/questions_vecs", bert_vecs)


def get_tfidf_arr():
    tv = TfidfVectorizer(min_df=2)
    stemmer = SnowballStemmer("english")

    with open_file("data/pure_questions.txt") as f:
        lines = f.readlines()
    print(len(lines))

    final_lines = []
    for i in range(len(lines)):
        if i % 1000 == 0:
            print(str(i) + " is processing...")
        line = lines[i]

        # 1. 转小写
        lower = line.strip().lower()
        # 2. 去标点
        remove_punc = lower.translate(str.maketrans("", "", string.punctuation))
        # 3. 分词
        tokens = nltk.word_tokenize(remove_punc)
        # 4. 去除停用词
        remove_sw = [w for w in tokens if w not in stopwords.words('english')]
        # 5. 次干提取
        stemmed = [stemmer.stem(w) for w in remove_sw]
        final_lines.append(" ".join(stemmed))

    tv_fit = tv.fit_transform(final_lines)
    tfidf_arr = tv_fit.toarray()
    print(tfidf_arr.shape)
    # TODO: PCA
    np.save("data/tfidf_arr", tfidf_arr)


def get_bert_train_test():
    lines_ = []
    with open_file("data/all_result.txt") as f:
        lines = f.readlines()
    print(len(lines))
    for line in lines:
        if len(line.strip().split("\t")) == 4:
            label, _, query, _ = line.strip().lower().split("\t")
        else:
            label, query = line.strip().lower().split("\t")
        lines_.append(label + "\t" + query + "\n")
    random_sample(lines_)


def get_bert_sent_vec():
    bv_res = []

    lines = read_file("data/bert_vecs.txt")
    for i in range(len(lines)):
        line = lines[i]

        jl = json.loads(line)
        bv = [lv["values"] for lv in jl["features"][0]["layers"]]
        bv_mean = np.mean(bv, axis=0).tolist()
        bv_res.append(bv_mean)

    print(len(bv_res))
    print(bv_res[0])
    print(len(bv_res[0]))

    np.save("data/bert_arr", bv_res)


def process_bank_ori_data():
    new_lines = []
    clz_ori = "".join(open("data/bank_ori_data.txt").readlines()).split("\"\n\"")
    for i in range(len(clz_ori)):
        label = 366 + i
        for q in re.split("\\||\n", clz_ori[i]):
            if q.strip() != "":
                new_lines.append(str(label) + "\t" + q.strip())
    write_lines("data/bank_result.txt", new_lines)


# process_ori_data()
# bc = BertClient()
# sent_to_vec()
# get_tfidf_arr()
# get_bert_train_test()
get_bert_sent_vec()
# process_bank_ori_data()

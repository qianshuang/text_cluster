# -*- coding: utf-8 -*-

from common import *
import string
from bert_serving.client import BertClient
import nltk
from nltk.corpus import stopwords


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
        remove_sw = [w for w in tokens if not w in stopwords.words('english')]

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


process_ori_data()
bc = BertClient()
# sent_to_vec()

# -*- coding: utf-8 -*-

from common import *
from Louvain import *

simi_threshold = 0.9

questions_vecs = np.load("data/bert_arr.npy")
with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

G_ = collections.defaultdict(dict)

for i in range(len(questions_vecs)):
    print(i, " is processing...")
    vi = questions_vecs[i]
    for j in range(i + 1, len(questions_vecs)):
        vj = questions_vecs[j]
        cos_simi = get_cos_similar(vi, vj)
        if cos_simi >= simi_threshold:
            G_[i][j] = cos_simi

algorithm = Louvain(G_)
communities = algorithm.execute()

# 按照社区大小从大到小排序输出
communities = sorted(communities, key=lambda b: -len(b))
write_res("res_bert_louvain", communities, pure_questions, 5)

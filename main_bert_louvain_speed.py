# -*- coding: utf-8 -*-

from common import *
from Louvain import *
from sklearn.metrics.pairwise import cosine_similarity

simi_threshold = 0.9

questions_vecs = np.load("data/bert_arr.npy")
with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

G_ = collections.defaultdict(dict)

print("cosine_similarity computing...")
cos_simi_m = cosine_similarity(questions_vecs)

print("graph building...")
for i in range(len(questions_vecs)):
    if i % 1000 == 0:
        print(i, "is processing...")

    for j in range(i + 1, len(questions_vecs)):
        if cos_simi_m[i][j] >= simi_threshold:
            G_[i][j] = cos_simi_m[i][j]
            G_[j][i] = cos_simi_m[j][i]

print("Louvain building...")
algorithm = Louvain(G_)

print("Louvain computing...")
communities = algorithm.execute()

# 按照社区大小从大到小排序输出
communities = sorted(communities, key=lambda b: -len(b))
write_res("res_bert_louvain_speed", communities, pure_questions, 5)

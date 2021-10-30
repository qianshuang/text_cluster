# -*- coding: utf-8 -*-

from common import *
import networkx as nx
import community as community_louvain
from sklearn.metrics.pairwise import cosine_similarity

questions_vecs = np.load("data/test_bert_arr.npy")
with open_file("data/test_questions.txt") as f:
    pure_questions = f.readlines()

print("cosine_similarity computing...")
cos_simi_m = cosine_similarity(questions_vecs)

for ep in range(80, 101):
    simi_threshold = ep / 100
    print(ep, "is processing...")

    G = nx.Graph()
    for i in range(len(questions_vecs)):
        if i % 1000 == 0:
            print(i, "is processing...")

        for j in range(i + 1, len(questions_vecs)):
            if cos_simi_m[i][j] >= simi_threshold:
                G.add_edge(i, j, weight=cos_simi_m[i][j])

    communities = community_louvain.best_partition(G)

    predict_dict = {}
    for k, v in communities.items():
        predict_dict.setdefault(v, set())
        predict_dict[v].add(k)
    predict_dict = list(predict_dict.values())

    # 按照社区大小从大到小排序输出
    communities = sorted(predict_dict, key=lambda b: -len(b))
    write_res("res_bert_louvain_speed/res_" + str(ep), communities, pure_questions, 5)

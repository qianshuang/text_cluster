# -*- coding: utf-8 -*-

from common import *
from Louvain import *
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

simi_threshold = 0.9

questions_vecs = np.load("data/bert_arr.npy")
with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

print("cosine_similarity computing...")
cos_simi_m = cosine_similarity(questions_vecs)

print("graph building...")
G = nx.Graph()
for i in range(len(questions_vecs)):
    if i % 1000 == 0:
        print(i, "is processing...")

    for j in range(i + 1, len(questions_vecs)):
        if cos_simi_m[i][j] >= simi_threshold:
            G.add_edge(i, j, weight=cos_simi_m[i][j])

communities = nx.connected_components(G)
communities = sorted(communities, key=lambda b: -len(b))  # 从大到小排序
write_res("res_max_conn_graph", communities, pure_questions, 5)

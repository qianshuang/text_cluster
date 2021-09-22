# -*- coding: utf-8 -*-

from common import *
import networkx as nx

with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

clusters_ = np.load("data/cluster_metric.npy", allow_pickle=True)
clusters_ = [list(i) for i in clusters_]

# build graph
G = nx.Graph()
for i in range(len(clusters_)):
    link = clusters_[i]
    if len(link) > 1:
        for j in link:
            if i != j:
                # G.add_edge(i, j, get_cos_similar(questions_vecs[i], questions_vecs[j])) # 推荐
                G.add_edge(i, j)

communities = nx.connected_components(G)
communities = sorted(communities, key=lambda b: -len(b))  # 从大到小排序
write_res("res_max_conn_graph", communities, pure_questions, 6)

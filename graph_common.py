# -*- coding: utf-8 -*-

import networkx as nx

simi_threshold = 0.9


def build_graph(questions_vecs, cos_simi_m):
    G = nx.Graph()
    for i in range(len(questions_vecs)):
        if i % 1000 == 0:
            print(i, "is building graph...")

        for j in range(i + 1, len(questions_vecs)):
            if cos_simi_m[i][j] >= simi_threshold:
                G.add_edge(i, j, weight=cos_simi_m[i][j])
    return G


def comm_dic_to_clusters(communities):
    predict_dict = {}
    for k, v in communities.items():
        predict_dict.setdefault(v, set())
        predict_dict[v].add(k)
    predict_dict = list(predict_dict.values())
    return predict_dict

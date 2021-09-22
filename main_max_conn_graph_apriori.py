# -*- coding: utf-8 -*-

import itertools

from common import *
import networkx as nx
from efficient_apriori import apriori

with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

clusters_ = np.load("data/cluster_metric.npy", allow_pickle=True)
print(clusters_)
print(len(clusters_))
print("------------")

clusters__ = []
for i in clusters_:
    if len(i) >= 2 and set(i) not in clusters__:
        clusters__.append(set(i))
clusters_ = clusters__
print(clusters_)
print(len(clusters_))
print("------------")

itemsets, rules = apriori(clusters_, min_support=1E-4, min_confidence=1E-4)
# print(rules)
print(len(rules))
print("------------")

# 融合：最大连通图
G = nx.Graph()
clusters_ = []
for rule in rules:
    l_ = list(rule.lhs)
    r_ = list(rule.rhs)
    for i in list(itertools.combinations(l_, 2)):
        G.add_edge(list(i)[0], list(i)[1])
    for i in list(itertools.combinations(r_, 2)):
        G.add_edge(list(i)[0], list(i)[1])
    G.add_edge(l_[0], r_[0])
communities = nx.connected_components(G)
print(communities)

# 按照社区大小从大到小排序输出
communities = sorted(clusters_, key=lambda b: -len(b))
write_res("res_max_conn_graph_apriori", communities, pure_questions, 2)

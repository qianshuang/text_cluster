# -*- coding: utf-8 -*-

import numpy as np
from efficient_apriori import apriori
import time
import itertools

# from bert_serving.client import BertClient
# from common import *
#
# lines = read_file("data/test.txt")
# bc = BertClient()
# bert_vecs = bc.encode(lines)
#
#
# def get_cos_similar(v1, v2):
#     num = float(np.dot(v1, v2))  # 向量点乘
#     denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
#     return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
#
#
# def n_largest(arr, n):
#     max_number = heapq.nlargest(n, arr)
#     max_index = []
#     for t in max_number:
#         index = arr.index(t)
#         max_index.append(index)
#         arr[index] = 0
#     return dict(zip(max_index, max_number))
#
#
# def most_common(row, n):
#     tar = bert_vecs[row]
#     cos_sims = [get_cos_similar(tar, i) for i in bert_vecs]
#     idx_sco_dic = n_largest(cos_sims, n)
#     for idx, sco in idx_sco_dic.items():
#         print({lines[idx]: sco})
#
#
# most_common(2, 3)


# import networkx as nx
#
# pointList = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
# linkList = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('E', 'F'), ('F', 'G')]
# G = nx.Graph()
#
# for node in pointList:
#     G.add_node(node)
#
# for link in linkList:
#     G.add_edge(link[0], link[1])
#
# for c in nx.connected_components(G):
#     print(c)

print("start k-means cluster for", 5, "...")

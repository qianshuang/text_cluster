# -*- coding: utf-8 -*-

from Louvain import *
from common import *

with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

# questions_vecs = np.load("data/questions_vecs.npy", allow_pickle=True)
clusters_ = np.load("data/cluster_metric.npy", allow_pickle=True)
clusters_ = [list(i) for i in clusters_]

G = load_graph(clusters_)
# G = load_graph(clusters_, questions_vecs)  # 推荐
algorithm = Louvain(G)
communities = algorithm.execute()

# 按照社区大小从大到小排序输出
communities = sorted(communities, key=lambda b: -len(b))
write_res("res_louvain", communities, pure_questions, 6)

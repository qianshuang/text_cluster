# -*- coding: utf-8 -*-

from common import *
from efficient_apriori import apriori

# questions_vecs = np.load("data/questions_vecs.npy")
with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

# def most_common(row, n):
#     tar = questions_vecs[row]
#     cos_sims = [get_cos_similar(tar, i) for i in questions_vecs]
#     idx_sco_dic = n_largest(cos_sims, n)
#     for idx, sco in idx_sco_dic.items():
#         print({pure_questions[idx]: sco})


# def cluster_():
#     clusters_ = []
#     for i in range(len(questions_vecs)):
#         print(str(i) + " is processing...")
#         cluster_ = []
#         for j in range(len(questions_vecs)):
#             cos_sim = get_cos_similar(questions_vecs[i], questions_vecs[j])
#             if cos_sim >= 0.99:
#                 cluster_.append(j)
#         clusters_.append(cluster_)
#
#     np.save("data/cluster_metric", clusters_)


clusters_ = np.load("data/cluster_metric.npy", allow_pickle=True)
clusters__ = []
for i in clusters_:
    if len(i) >= 2 and set(i) not in clusters__:
        clusters__.append(set(i))
clusters_ = clusters__
print(len(clusters_))

itemsets, rules = apriori(clusters_, min_support=1E-4, min_confidence=1E-4)
clusters_ = []
for rule in rules:
    u_set = set(rule.lhs).union(set(rule.rhs))
    if u_set not in clusters_:
        clusters_.append(u_set)
print(len(clusters_))

# 融合
for i in range(len(clusters_) - 1):
    i_0 = clusters_[i]
    for j in range(i + 1, len(clusters_)):
        i_1 = clusters_[j]
        if len(i_0.intersection(i_1)) >= len(i_1) / 3:  # 交集超过1/3就融合
            clusters_[i] = i_0.union(i_1)
            clusters_[j] = {}

communities = [i for i in clusters_ if i != {}]
print(len(communities))
communities = sorted(communities, key=lambda b: -len(b))  # 从大到小排序
write_res("res_main", communities, pure_questions, 6)

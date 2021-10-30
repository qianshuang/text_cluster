# -*- coding: utf-8 -*-

from sklearn.cluster import SpectralClustering
from common import *
from sklearn.metrics.pairwise import cosine_similarity

with open_file("data/test_questions.txt") as f:
    lines = f.readlines()
bert_arr = np.load("data/test_bert_arr.npy", allow_pickle=True)

print("cosine_similarity computing...")
# cos_simi_m = cosine_similarity(bert_arr)
cos_simi_m = cos_distance_metric(bert_arr)

y_pred = SpectralClustering(affinity="precomputed", n_clusters=12).fit_predict(cos_simi_m)
best_res = y_pred  # 返回所有簇id
best_k = np.max(y_pred)

clusters = []
for k in range(best_k):
    ids = [i for i, x in enumerate(best_res) if x == k]
    clusters.append(ids)

communities = sorted(clusters, key=lambda b: -len(b))  # 从大到小排序
write_res("res_bert_spectral", communities, lines, 5)

# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
from common import *
from sklearn import metrics

with open_file("data/pure_questions.txt") as f:
    lines = f.readlines()
bert_arr = np.load("data/tfidf_arr.npy", allow_pickle=True)

max_steps_without_improve = 5
best_k = 250
best_scr = 0
best_res = []
for i in range(250, 251):
    # early stopping
    if i - best_k >= max_steps_without_improve:
        break

    print("start k-means cluster for", i, "...")
    kmeans = KMeans(n_clusters=i).fit(bert_arr)
    score = metrics.calinski_harabasz_score(bert_arr, kmeans.predict(bert_arr))
    if score > best_scr:
        best_k = i
        best_scr = score
        best_res = kmeans.labels_  # 返回所有簇id
    print("score:", score)

print("best_k:", best_k)
print("best_scr:", best_scr)

clusters = []
for k in range(best_k):
    ids = [i for i, x in enumerate(best_res) if x == k]
    clusters.append(ids)

write_res("res_tfidf", clusters, lines, 5)

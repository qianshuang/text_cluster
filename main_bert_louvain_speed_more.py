# -*- coding: utf-8 -*-

from common import *
from sklearn.metrics.pairwise import cosine_similarity
from communities.algorithms import louvain_method

simi_threshold = 0.9

questions_vecs = np.load("data/bert_arr.npy")
with open_file("data/pure_questions.txt") as f:
    pure_questions = f.readlines()

print("cosine_similarity computing...")
cos_simi_m = cosine_similarity(questions_vecs)

communities, _ = louvain_method(cos_simi_m)

# 按照社区大小从大到小排序输出
communities = sorted(communities, key=lambda b: -len(b))
write_res("res_bert_louvain_speed_more", communities, pure_questions, 5)

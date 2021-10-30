# -*- coding: utf-8 -*-

from common import *
from bert_common import *
from graph_common import *
import tensorflow as tf

from sklearn.metrics.pairwise import cosine_similarity

import community as community_louvain

batch_size = 100

# 1. 加载模型
start = datetime.datetime.now()
predict_fn = tf.contrib.predictor.from_saved_model("export")
print("load model cost:", time_cost(start), "ms")

# 2.（分批执行）
start = datetime.datetime.now()
with open_file("data/test_questions.txt") as f:
    lines = f.readlines()
lines_arr = np.array_split(np.array(lines), int(len(lines) / batch_size) + 1)

all_res = []
for i in range(len(lines_arr)):
    print(i * batch_size, "is processing...")
    # 构建bert输入特征
    sub_lines = lines_arr[i]
    feed_dict = {"input_ids": [], "input_mask": [], "segment_ids": []}
    for line in sub_lines:
        input_ids, input_mask, segment_ids = convert_single_example(line.strip())
        feed_dict["input_ids"].append(input_ids)
        feed_dict["input_mask"].append(input_mask)
        feed_dict["segment_ids"].append(segment_ids)

    # 3. 得到bert句向量
    prediction = predict_fn(feed_dict)
    query_output = prediction["query_output"]
    all_res.append(query_output)

query_bert_arr = np.concatenate(all_res, axis=0)
print("gen query bert array cost:", time_cost(start), "ms")

# 4. 聚类
start = datetime.datetime.now()
cos_simi_m = cosine_similarity(query_bert_arr)
G = build_graph(query_bert_arr, cos_simi_m)
communities = community_louvain.best_partition(G)
communities = comm_dic_to_clusters(communities)
print("cluster cost:", time_cost(start), "ms")

# 5. 按照社区大小从大到小排序输出
start = datetime.datetime.now()
communities = sorted(communities, key=lambda b: -len(b))
write_res("res_bert_louvain_with_pak", communities, lines, 5)
print("output result cost:", time_cost(start), "ms")

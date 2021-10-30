# -*- coding: utf-8 -*-

import pandas as pd
import itertools
from tqdm import tqdm
import os

label_dict = {}
df = pd.read_excel('data/HE_test.xlsx')
for i, row in df.iterrows():
    label = row['category'].strip()
    sentence = row['data'].strip()
    label_dict.setdefault(label, set())
    label_dict[label].add(sentence)
print(label_dict)
labels = list(label_dict.keys())
p_labels = list(itertools.permutations(labels))

dirs_ = os.listdir("res_bert_louvain_speed")
for d_ in dirs_:
    th = d_.split("_")[1]
    predict_dict = {}
    dirs = os.listdir("res_bert_louvain_speed/" + d_)
    for file in dirs:
        index = int(file.split(".")[0])
        if index > 5:
            continue
        with open("res_bert_louvain_speed/" + d_ + "/" + file, encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            for line in lines:
                predict_dict.setdefault(index, set())
                predict_dict[index].add(line.strip())

    max_accurate = 0
    for p in tqdm(p_labels, total=len(p_labels)):
        n_correct = 0
        n_total = 0
        for i, label in enumerate(p):
            trues = label_dict[label]
            predicts = predict_dict[i]
            common = trues & predicts
            n_correct += len(common)
            n_total += len(predicts)
        accurate = n_correct / n_total

        if accurate > max_accurate:
            df_save = df.copy()
            max_accurate = accurate
            # 保存结果到excel
            for cluster_id, questions in predict_dict.items():
                for question in questions:
                    df_save.loc[df_save[df_save['data'] == question].index.tolist(), 'cluster_ids'] = cluster_id
            # print(p)
            # print(df_save['cluster_ids'].isnull().sum(axis=0))

    print(th, 'max_accurate:', max_accurate)
    # df_save.to_excel('data/bert_louvain_accurate_result.xlsx')

# -*- coding: utf-8 -*-

import nltk
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from common import *
import string
from nltk.corpus import stopwords

tv = TfidfVectorizer()
stemmer = SnowballStemmer("english")

with open_file("data/pure_questions.txt") as f:
    lines = f.readlines()
print(len(lines))

final_lines = []
for i in range(len(lines)):
    if i % 1000 == 0:
        print(str(i) + " is processing...")
    line = lines[i]

    # 1. 转小写
    lower = line.strip().lower()
    # 2. 去标点
    remove_punc = lower.translate(str.maketrans("", "", string.punctuation))
    # 3. 分词
    tokens = nltk.word_tokenize(remove_punc)
    # 4. 去除停用词
    remove_sw = [w for w in tokens if w not in stopwords.words('english')]
    # 5. 次干提取
    stemmed = [stemmer.stem(w) for w in remove_sw]
    final_lines.append(" ".join(stemmed))

tv_fit = tv.fit_transform(final_lines)
tfidf_arr = tv_fit.toarray()

best_k = 0
best_scr = 0
best_res = []
for i in range(5, 100):
    print("start k-means cluster for", i, "...")
    kmeans = KMeans(n_clusters=i).fit(tfidf_arr)
    score = metrics.calinski_harabaz_score(tfidf_arr, kmeans.predict(tfidf_arr))
    if score > best_scr:
        best_k = i
        best_scr = score
        best_res = kmeans.labels_  # 返回所有簇id

print("best_k:", best_k)
print("best_scr:", best_scr)

clusters = []
for k in range(best_k):
    ids = [i for i, x in enumerate(best_res) if x == k]
    clusters.append(ids)

write_res("res_tfidf", clusters, lines, 5)

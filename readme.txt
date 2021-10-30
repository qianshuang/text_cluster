data_process_final.py
    process_ori_data() --> 1. 清洗原始文件
    sent_to_vec() --> 2. 得到BERT句向量（这里直接使用原生bert模型，如果能标注几百条进行fine-tune，效果更好）

main.py
    cluster()  --> 3. 得到相似度矩阵（阈值可以自行调整），并使用apriori进行关联挖掘（系数可以调整）
    cluster_arr_()  --> 4. 进行聚类融合
    get_res()  --> 5. 得到最终结果

main_max_conn_graph.py：最大连通图
main_louvain.py：最大连通图分裂后的结果，应用louvain社区发现算法
main_max_conn_graph_apriori.py：先取最大联通子图，再做apriori
main_apriori_max_conn_graph.py：先做apriori，再取最大联通子图

tips:
* 最终结果生成在result目录下，可以按照步骤进行复现。初看正确率接近100%，由于时间紧迫，未进行任何模型调优，后续可以基于此代码进行优化，效果会更好。
* bert模型请下载：multi_cased_L-12_H-768_A-12，由于文件过大故不打包进源代码中。


export BERT_BASE_DIR=uncased_L-12_H-768_A-12

python3 extract_features.py \
  --input_file=../data/test_questions.txt \
  --output_file=../data/test_bert_vecs.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=output/model.ckpt-2313 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt

python3 run_classifier.py \
  --data_dir=../data \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --task_name=comm100 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --output_dir=output \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --do_eval=True \
  --num_train_epochs=1

python3 run_classifier.py \
  --data_dir=../data \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --task_name=comm100 \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --output_dir=output \
  --do_predict=True

python3 /root/anaconda3/envs/tf-gpu1.15-py3.6/lib/python3.6/site-packages/tensorflow_core/python/tools/saved_model_cli.py \
  show --dir export/temp-b'1635553889' --all
#!/usr/bin/env bash
python models/train_rels.py -m predcls -p 100 -clip 5 \
-tb_log_dir summaries/PredCls \
-save_dir checkpoints/PredCls \
-ckpt checkpoints/PredCls/vgrel-0.tar \
-val_size 5000 \
-adam \
-b 2 \
-lr 1e-5 \
-use_dualResGCN_rel \
-dualResGCN_rel_hidden_dim 512 \
-dualResGCN_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge lib/rel_matrix.npy
 

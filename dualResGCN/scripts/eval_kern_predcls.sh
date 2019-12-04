#!/usr/bin/env bash
python models/eval_rels.py -m predcls -p 100 -clip 5 \
-ckpt ./../../ResGCN_Objectness/checkpoints/kern_predcls_downsample14_topk70_removeF1trick_fineture/vgrel-13.tar \
-test \
-b 1 \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy \
-cache caches/kern_predcls_newbox.pkl \
-save_rel_recall results/kern_rel_recall_predcls.pkl

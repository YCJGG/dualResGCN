#!/usr/bin/env bash
python models/train_rels.py -m sgcls -p 100 -clip 5 \
-tb_log_dir summaries/kern_sgcls_no_degree \
-save_dir checkpoints/kern_sgcls_no_degree \
-ckpt checkpoints/kern_sgcls_fine_remove_old_Objlogits/vgrel-1.tar \
-val_size 5000 \
-adam \
-b 2 \
-lr 1e-6 \
-use_ggnn_obj \
-ggnn_obj_time_step_num 3 \
-ggnn_obj_hidden_dim 512 \
-ggnn_obj_output_dim 512 \
-use_obj_knowledge \
-obj_knowledge prior_matrices/obj_matrix.npy \
-use_ggnn_rel \
-ggnn_rel_time_step_num 3 \
-ggnn_rel_hidden_dim 512 \
-ggnn_rel_output_dim 512 \
-use_rel_knowledge \
-rel_knowledge prior_matrices/rel_matrix.npy

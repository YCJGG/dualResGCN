
from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
import torch
from torch.autograd import Variable
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from tqdm import tqdm
from config import BOX_SCALE, IM_SCALE
import dill as pkl
import os
from lib.dualResGCN_model import dualResGCN
import torch.nn.functional as F


conf = ModelConfig()


train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')
ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship
if conf.test:
    val = test
    #val =val
train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector_rel = dualResGCN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                use_dualResGCN_rel=conf.use_dualResGCN_rel,
                dualResGCN_rel_hidden_dim=conf.dualResGCN_rel_hidden_dim, dualResGCN_rel_output_dim=conf.dualResGCN_rel_output_dim,
                use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge)

print(detector_rel)
detector_rel.cuda()

print(conf.ckpt)

weights = torch.load(conf.ckpt)



optimistic_restore(detector_rel, weights)

torch.save(detector_rel.state_dict(), 'dualResGCN_PredCls.tar')
all_pred_entries = []

def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, thrs=(20, 50, 100)):
    det_res = detector_rel[b]
    
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, dists_i, rel_inds, save_entry) in enumerate(det_res):

        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        
        assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)

        
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,
        }
       
        all_pred_entries.append(pred_entry)
        
        test_entry = {
            'gt_entry': gt_entry,
            'save_entry': save_entry
        }

        
        
        
    eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, 
               evaluator_list, evaluator_multiple_preds_list)




evaluator = BasicSceneGraphEvaluator.all_modes()
evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
evaluator_list = [] # for calculating recall of each relationship except no relationship
evaluator_multiple_preds_list = []

for index, name in enumerate(ind_to_predicates):
    if index == 0:
        continue
    evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
    evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))


detector_rel.eval()
for val_b, batch in enumerate(tqdm(val_loader)):
#     if val_b == 1000:
#         break
    val_batch(conf.num_gpus*val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)
recall = evaluator[conf.mode].print_stats()
recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True, save_file=conf.save_rel_recall)



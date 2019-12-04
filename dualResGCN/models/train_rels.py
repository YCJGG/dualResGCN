"""
Training script for scene graph detection. Integrated with Rowan's faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import topkloss
from topk_crossEntrophy import topk_crossEntrophy
# import KERN model
from lib.dualResGCN_model import dualResGCN
from tqdm import tqdm
import dill as pkl
from torch.autograd import Variable
import math


conf = ModelConfig()

# We use tensorboard to observe results and decrease learning rate manually. If you want to use TB, you need to install TensorFlow fist.
if conf.tb_log_dir is not None:
    from tensorboardX import SummaryWriter
    if not os.path.exists(conf.tb_log_dir):
        os.makedirs(conf.tb_log_dir) 
    writer = SummaryWriter(log_dir=conf.tb_log_dir)
    use_tb = True
else:
    use_tb = False

train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet')

ind_to_predicates = train.ind_to_predicates # ind_to_predicates[0] means no relationship

val = test

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus)

detector = dualResGCN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                use_ggnn_obj=conf.use_ggnn_obj, ggnn_obj_time_step_num=conf.ggnn_obj_time_step_num,
                ggnn_obj_hidden_dim=conf.ggnn_obj_hidden_dim, ggnn_obj_output_dim=conf.ggnn_obj_output_dim,
                use_obj_knowledge=conf.use_obj_knowledge, obj_knowledge=conf.obj_knowledge,
                use_dualResGCN_rel=conf.use_dualResGCN_rel,
                dualResGCN_rel_hidden_dim=conf.dualResGCN_rel_hidden_dim, dualResGCN_rel_output_dim=conf.dualResGCN_rel_output_dim,
                use_rel_knowledge=conf.use_rel_knowledge, rel_knowledge=conf.rel_knowledge)

# Freeze the detector
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

print(print_para(detector), flush=True)


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.
    fc_params = [p for n,p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
    obj_params = [p for n,p in detector.named_parameters() if n.startswith('ggnn_obj_reason') and p.requires_grad]
    

    non_fc_params = [p for n,p in detector.named_parameters() if not n.startswith('roi_fmap') and not n.startswith('ggnn_obj_reason') and p.requires_grad]
    
  
    
    
    params = [{'params': fc_params, 'lr': lr / 10.0},{'params': obj_params, 'lr': lr  }, {'params': non_fc_params, 'lr': lr}]
  
    #params = [p for n,p in detector.named_parameters() if p.requires_grad]
  
    #if conf.adam:
    optimizer = optim.Adam(params, weight_decay=conf.adamwd, lr=lr, eps=1e-3)
    #else:
    #    optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    #scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
    #                               verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer # , scheduler


ckpt = torch.load(conf.ckpt)
if conf.ckpt.split('-')[-2].split('/')[-1] == 'vgrel':
    print("Loading EVERYTHING")
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
        # optimistic_restore(detector.detector, torch.load('checkpoints/vgdet/vg-28.tar')['state_dict'])
else:
    start_epoch = -1
    optimistic_restore(detector.detector, ckpt['state_dict'])

    detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
    detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
    detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
    detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

# cls_ckpt = torch.load('./SGDet_Model/kern_sgdet.tar')['state_dict']

# # # print(cls_ckpt.keys())
# # #load classifier

# detector.ggnn_obj_reason.ggnn_obj.fc_eq3_u.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq3_u.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq3_u.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq3_u.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq3_w.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq3_w.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq4_w.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq4_w.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq4_w.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq4_w.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_obj_cls.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_obj_cls.weight'])
# detector.ggnn_obj_reason.obj_proj.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.obj_proj.weight'])
# detector.ggnn_obj_reason.obj_proj.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.obj_proj.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq5_w.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq5_w.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq5_u.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq5_u.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq4_u.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq4_u.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_output.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_output.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq5_w.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq5_w.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq5_u.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq5_u.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq4_u.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq4_u.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_output.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_output.weight'])
# detector.ggnn_obj_reason.ggnn_obj.fc_obj_cls.bias.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_obj_cls.bias'])
# detector.ggnn_obj_reason.ggnn_obj.fc_eq3_w.weight.data.copy_(cls_ckpt ['ggnn_obj_reason.ggnn_obj.fc_eq3_w.weight'])


# state_dict = detector.state_dict()

# state_dict['roi_fmap_obj_cls.0.weight'] = cls_ckpt ['roi_fmap_obj.0.weight']
# state_dict['roi_fmap_obj_cls.0.bias'] = cls_ckpt ['roi_fmap_obj.0.bias']
# state_dict['roi_fmap_obj_cls.3.weight'] = cls_ckpt ['roi_fmap_obj.3.weight']
# state_dict['roi_fmap_obj_cls.3.bias'] = cls_ckpt ['roi_fmap_obj.3.bias']

# detector.load_state_dict(state_dict)  
    
    
    
    

detector.cuda()
    


topk_crossEntropy = topk_crossEntrophy()



def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()
    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, b, verbose=b % (conf.print_interval*10) == 0)) 
        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)

def degree_count_(result):
    
    labels_obj_count = result.rel_labels[:,1:].clone()
    obj_labels = result.rm_obj_labels.clone()

    labels_obj_count = torch.cat((obj_labels[labels_obj_count[:, 0]].view(-1, 1), obj_labels[labels_obj_count[:, 1]].view(-1, 1)), 1)
    
    index = torch.nonzero(labels_obj_count[:,-1])
    
    if len(index) == 0:
        return 1
    
    index = index.view(-1) 
    labels_obj_count = torch.index_select(labels_obj_count, 0 , index)
    labels_obj_count = labels_obj_count[:,:-1]
    
    labels_obj_count = labels_obj_count.data.cpu().numpy().squeeze().tolist()
    

    
    degree_count = []

    for i in range(result.rm_obj_labels.size(0)):
        degree_count += [labels_obj_count.count(result.rm_obj_labels[i].data.cpu().numpy())]
    degree_count = [(x+1) / math.sqrt(x+1) for x in degree_count]

    
    degree_count = np.array(degree_count, dtype = np.float32)
    
    degree_count = torch.from_numpy(degree_count)
    degree_count = Variable(degree_count.cuda())
    degree_count = degree_count.cuda()
    
    return degree_count


def train_batch(b, index, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
   
    result = detector[b]
    
    degree_count = degree_count_(result)

    losses = {}
    if conf.use_ggnn_obj: # if not use ggnn obj, we just use scores of faster rcnn as their scores, there is no need to train
        obj_loss = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels) 
        #losses['class_loss'] = torch.mean(obj_loss * degree_count)
        losses['class_loss'] = obj_loss
       
    
    labels_binary = result.rel_labels[:,-1].clone()
  
    labels_binary[labels_binary != 0 ] = 1
    
    labels_binary = labels_binary.float()
    
    losses['bce_loss'] = F.binary_cross_entropy_with_logits(result.rel_dists[1], labels_binary.view(-1,1))
 
  
    losses['rel_loss'] = topk_crossEntropy(result.rel_dists[0], result.rel_labels[:, -1], top_k = 0.7)
    
 
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.data[0] for x, y in losses.items()})
    return res 


def val_epoch():
    detector.eval()
    evaluator_list = [] # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))
    evaluator = BasicSceneGraphEvaluator.all_modes() # for calculating recall
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    for val_b, batch in enumerate(tqdm(val_loader)):
        
        val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list)
#         if val_b == 0:
#             break
    recall = evaluator[conf.mode].print_stats()
    recall_mp = evaluator_multiple_preds[conf.mode].print_stats()
    
    mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode)
    mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True)

    return recall, recall_mp, mean_recall, mean_recall_mp


def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i, dists_i, rel_inds,_) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)
    
        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds, 
                   evaluator_list, evaluator_multiple_preds_list)

print("Training starts now!")
optimizer = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):
#     if epoch > 1: 
    rez = train_epoch(epoch)
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    if use_tb:
        writer.add_scalar('loss/rel_loss', rez.mean(1)['rel_loss'], epoch)
        if conf.use_ggnn_obj:
            writer.add_scalar('loss/class_loss', rez.mean(1)['class_loss'], epoch)
        writer.add_scalar('loss/total', rez.mean(1)['total'], epoch)

    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(), #{k:v for k,v in detector.state_dict().items() if not k.startswith('detector.')},
            # 'optimizer': optimizer.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    recall, recall_mp, mean_recall, mean_recall_mp = val_epoch()
    if use_tb:
        for key, value in recall.items():
            writer.add_scalar('eval_' + conf.mode + '_with_constraint/' + key, value, epoch)
        for key, value in recall_mp.items():
            writer.add_scalar('eval_' + conf.mode + '_without_constraint/' + key, value, epoch)
        for key, value in mean_recall.items():
            writer.add_scalar('eval_' + conf.mode + '_with_constraint/mean ' + key, value, epoch)
        for key, value in mean_recall_mp.items():
            writer.add_scalar('eval_' + conf.mode + '_without_constraint/mean ' + key, value, epoch)



import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math



class ResBlock2(nn.Module):
    def __init__(self, input_channel, output_channel,  bias=True):
        super(ResBlock2,self).__init__()
        self.input_channel= input_channel
        self.output_channel = output_channel
        

        self.WG_1 = nn.Linear(22, 256,bias = False)
        self.WK_1 = nn.Linear(512,256, bias = False)
        self.WQ_1 = nn.Linear(512,256, bias = False)
        self.WV_1 = nn.Linear(512,256, bias = False)
    
        self.WG_2 = nn.Linear(22, 256,bias = False)
        self.WK_2 = nn.Linear(512,256, bias = False)
        self.WQ_2 = nn.Linear(512,256, bias = False)
        self.WV_2 = nn.Linear(512,256, bias = False)



    def forward(self, input_x, box):
        
        visual_features_embedK1 = self.WK_1(input_x)
        visual_features_embedQ1 = self.WQ_1(input_x)
        WA1 = visual_features_embedK1.mm(visual_features_embedQ1.t())/16.0
        
        box_features_embed1 = self.WG_1(box)
        WG1 = F.relu(box_features_embed1.mm(box_features_embed1.t())/16.0)
        
        W1 = ( WG1 * torch.exp(WA1) ) / torch.sum(WG1 * torch.exp(WA1))
                
        FR1 = W1.mm(self.WV_1(input_x))
    
        visual_features_embedQ2 = self.WQ_2(input_x)
        visual_features_embedK2 = self.WK_2(input_x)
        WA2 = visual_features_embedK2.mm(visual_features_embedQ2.t())/16.0

        box_features_embed2 = self.WG_2(box)
        WG2 = F.relu(box_features_embed2.mm(box_features_embed2.t())/16.0)
        
        W2 = ( WG2 * torch.exp(WA2) ) / torch.sum(WG2 * torch.exp(WA2))
        FR2 = W2.mm(self.WV_2(input_x))

        FR = torch.cat((FR1,FR2),1)
        return FR*0.1 + input_x

class ResGCN(nn.Module):
    def __init__(self, num_rel_cls=51, hidden_dim=512, output_dim=512, num_layers = 18):
        super(ResGCN,self).__init__()
        self.num_rel_cls = num_rel_cls
        self.num_layers = num_layers

        self.Resgcn1 = ResBlock2(512,512)
        self.Resgcn2 = ResBlock2(512,512)
        self.Resgcn3 = ResBlock2(512,512)
        self.Resgcn4 = ResBlock2(512,512)
        self.Resgcn5 = ResBlock2(512,512)
        self.Resgcn6 = ResBlock2(512,512)
        self.Resgcn7 = ResBlock2(512,512)
        self.Resgcn8 = ResBlock2(512,512)
        self.Resgcn9 = ResBlock2(512,512)
        
        self.matrix = np.load('./lib/rel_matrix.npy').astype(np.float32)

        self.matrix_binary = np.load('./lib/rel_binary_matrix.npy').astype(np.float32)
        
        self.words = np.load('./data/word_embedding.npy')

        self.fc_emd_1 = nn.Linear(512*3, hidden_dim)
        self.fc_emd_2 = nn.Linear(hidden_dim+400, hidden_dim)

        self.fc_cls = nn.Linear(output_dim, self.num_rel_cls )  
        self.fc_o1 = nn.Linear(512+200, hidden_dim)
        self.fc_o2 = nn.Linear(512+200, hidden_dim)
        
        self.fc_o1_cls = nn.Linear(512,51)
        self.fc_o2_cls = nn.Linear(512,51)
        
      
        self.fc_binary_cls = nn.Linear(512,1)


        # box 
        self.fc_box = nn.Linear(22,64)
        self.fc_box_cls = nn.Linear(64,51)



    def forward(self, rel_inds, sub_obj_preds, input_ggnn, box_features):
        (input_rel_num, node_num, _) = input_ggnn.size()
        
        assert input_rel_num == len(rel_inds)

        batch_in_matrix_sub = np.zeros((input_rel_num, self.num_rel_cls), dtype=np.float32)
        batch_in_matrix_binary = np.zeros((input_rel_num, 1), dtype=np.float32)
        for index, rel in enumerate(rel_inds):
            batch_in_matrix_sub[index] = \
                self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
            batch_in_matrix_binary[index] = \
                self.matrix_binary[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data][0,1]
            
        batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False).cuda()
        batch_in_matrix_binary_gpu = Variable(torch.from_numpy(batch_in_matrix_binary), requires_grad=False).cuda()
        
        del batch_in_matrix_sub
        del batch_in_matrix_binary
        
        
        input_rel_num = input_ggnn.size(0)

        o1_feat = input_ggnn[:,:,0:512].contiguous()
        o2_feat = input_ggnn[:,:,512:1024].contiguous()

        input_ggnn = self.fc_emd_1(input_ggnn.view(input_rel_num,-1))
        
        o1_words = np.zeros((input_rel_num, 200), dtype = np.float32)
        o2_words = np.zeros((input_rel_num, 200), dtype = np.float32)
        for index, rel in enumerate(rel_inds):
            o1_words[index] = self.words[sub_obj_preds[index,0].cpu().data - 1]
            o2_words[index] = self.words[sub_obj_preds[index,1].cpu().data - 1]
        
        o1_words = Variable(torch.from_numpy(o1_words), requires_grad=False).cuda()
        o2_words = Variable(torch.from_numpy(o2_words), requires_grad=False).cuda()
        
        o1_feat_words = torch.cat((o1_feat.view(input_rel_num,-1), o1_words), 1)
        o2_feat_words = torch.cat((o2_feat.view(input_rel_num, -1), o2_words), 1)
        input_ggnn_words = torch.cat((o1_words, input_ggnn, o2_words), 1)

        
        gcn_fc = F.relu(self.fc_emd_2(input_ggnn_words.view(input_rel_num,-1)))
        
        gcn_fc = self.Resgcn1(gcn_fc, box_features)
        gcn_fc = self.Resgcn2(gcn_fc, box_features)
        gcn_fc = self.Resgcn3(gcn_fc, box_features)
        gcn_fc = self.Resgcn4(gcn_fc, box_features)
        gcn_fc = self.Resgcn5(gcn_fc, box_features)
        gcn_fc = self.Resgcn6(gcn_fc, box_features)
        gcn_fc = self.Resgcn7(gcn_fc, box_features)
        gcn_fc = self.Resgcn8(gcn_fc, box_features)
        gcn_fc = self.Resgcn9(gcn_fc, box_features)
           
        rel_dists1 = self.fc_cls(gcn_fc)
        o1_feature = F.relu(self.fc_o1(o1_feat_words))
        o1_dists = self.fc_o1_cls(o1_feature)
        o2_feature = F.relu(self.fc_o2(o2_feat_words))
        o2_dists = self.fc_o2_cls(o2_feature)
        
        #box
        box_feature_ = F.relu(self.fc_box(box_features))
        box_dist = self.fc_box_cls(box_feature_)
    
  
        rel_dists = rel_dists1 + batch_in_matrix_sub_gpu*5 + o1_dists + o2_dists + box_dist
        
        # binary trick. This probably doesn't help it much

        rel_binary_dists = self.fc_binary_cls(gcn_fc)

        rel_binary_dists = rel_binary_dists + batch_in_matrix_binary_gpu

        
        return [rel_dists, rel_binary_dists,rel_dists1, batch_in_matrix_sub_gpu, o1_dists, o2_dists, box_dist ]
   

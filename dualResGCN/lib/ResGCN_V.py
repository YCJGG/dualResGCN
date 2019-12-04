
import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class ResBlock(nn.Module):
	def __init__(self, input_channel, output_channel,  bias=True):
		super(ResBlock,self).__init__()
		self.input_channel= input_channel
		self.output_channel = output_channel
		

		self.WG_1 = nn.Linear(4, 256,bias = False)
		self.WK_1 = nn.Linear(512,256, bias = False)
		self.WQ_1 = nn.Linear(512,256, bias = False)
		self.WV_1 = nn.Linear(512,256, bias = False)
	
		self.WG_2 = nn.Linear(4, 256,bias = False)
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


class ResGCN_V(nn.Module):
	def __init__(self, num_rel_cls=51, hidden_dim=512, output_dim=512):
		super(ResGCN_V,self).__init__()
		self.num_rel_cls = num_rel_cls

		self.Resgcn1 = ResBlock(512,512)
		self.Resgcn2 = ResBlock(512,512)
		self.Resgcn3 = ResBlock(512,512)
		self.Resgcn4 = ResBlock(512,512)
		self.Resgcn5 = ResBlock(512,512)
		self.Resgcn6 = ResBlock(512,512)
		self.Resgcn7 = ResBlock(512,512)
		self.Resgcn8 = ResBlock(512,512)
		self.Resgcn9 = ResBlock(512,512)
		

		self.fc_emd_2 = nn.Linear(hidden_dim + 200, hidden_dim)
		self.fc_emd_3 = nn.Linear(output_dim *2 , hidden_dim)
		self.fc_cls = nn.Linear(output_dim, self.num_rel_cls )	
		
		self.words = np.load('./data/word_embedding.npy')

	def forward(self, rel_inds, sub_obj_preds, obj_preds, input_ggnn, box_features):
		(input_rel_num, node_num) = input_ggnn.size()
		
		box_features_ = torch.zeros((input_rel_num, 4))

		box_features = box_features / 592.0
		box_features_[:,0] = (box_features[:,0] + box_features[:,2])/2.0
		box_features_[:,1] = (box_features[:,1] + box_features[:,3] )/2.0
		box_features_[:,2] = box_features[:,2] - box_features[:,0] 
		box_features_[:,3] = box_features[:,3] - box_features[:,1] 
		
		box_features = Variable(box_features_, requires_grad = False).cuda()

		
		o_words = np.zeros((input_rel_num, 200), dtype = np.float32)
		for index in range(input_rel_num):
			o_words[index] = self.words[obj_preds[index].cpu().data - 1]
		o_words = Variable(torch.from_numpy(o_words), requires_grad=False).cuda()

		gcn = torch.cat((input_ggnn.view(input_rel_num,-1), o_words), 1)
		
		gcn = F.relu(self.fc_emd_2(gcn))
		
		gcn = self.Resgcn1(gcn, box_features) 
		gcn = self.Resgcn2(gcn, box_features) 
		gcn = self.Resgcn3(gcn, box_features) 
		gcn = self.Resgcn4(gcn, box_features) 
		gcn = self.Resgcn5(gcn, box_features) 
		gcn = self.Resgcn6(gcn, box_features) 
		gcn = self.Resgcn7(gcn, box_features) 
		gcn = self.Resgcn8(gcn, box_features) 
		gcn = self.Resgcn9(gcn, box_features) 
		
		gcn_fc = torch.stack([torch.cat([gcn[rel_ind[1]], gcn[rel_ind[2]]], 0)  for index, rel_ind in enumerate(rel_inds)])
 		
		gcn_fc = F.relu(self.fc_emd_3(gcn_fc))		

		rel_dists = self.fc_cls(gcn_fc)

		return rel_dists

# Well, this file contains modules of GGNN_obj and GGNN_rel
import os, sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class GGNNObj(nn.Module):
	def __init__(self, num_obj_cls=151, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, prior_matrix=''):
		super(GGNNObj, self).__init__()
		self.num_obj_cls = num_obj_cls
		self.time_step_num = time_step_num
		self.output_dim = output_dim

		if use_knowledge:
			matrix_np = np.load('./prior_matrices/obj_matrix.npy').astype(np.float32)
		else:
			matrix_np = np.ones((num_obj_cls, num_obj_cls)).astype(np.float32) / num_obj_cls

		self.matrix = Variable(torch.from_numpy(matrix_np), requires_grad=False).cuda()
		# if you want to use multi gpu to run this model, then you need to use the following line code to replace the last line code.
		# And if you use this line code, the model will save prior matrix as parameters in saved models.
		# self.matrix = nn.Parameter(torch.from_numpy(matrix_np), requires_grad=False)



		# here we follow the paper "Gated graph sequence neural networks" to implement GGNN, so eq3 means equation 3 in this paper.
		self.fc_eq3_w = nn.Linear(2*hidden_dim, hidden_dim)
		self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
		self.fc_eq4_w = nn.Linear(2*hidden_dim, hidden_dim)
		self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
		self.fc_eq5_w = nn.Linear(2*hidden_dim, hidden_dim)
		self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

		self.fc_output = nn.Linear(2*hidden_dim, output_dim)
		self.ReLU = nn.ReLU(True)
		self.fc_obj_cls = nn.Linear(self.num_obj_cls * output_dim, self.num_obj_cls)
		

	def forward(self, input_ggnn):
		# propogation process
		num_object = input_ggnn.size()[0]
		hidden = input_ggnn.repeat(1, self.num_obj_cls).view(num_object, self.num_obj_cls, -1)
		for t in range(self.time_step_num):
			# eq(2)
			# here we use some matrix operation skills
			hidden_sum = torch.sum(hidden, 0)
			av = torch.cat([torch.cat([self.matrix.transpose(0, 1) @ (hidden_sum - hidden_i) for hidden_i in hidden], 0),
						   torch.cat([self.matrix @ (hidden_sum - hidden_i) for hidden_i in hidden], 0)], 1)

			# eq(3)
			hidden = hidden.view(num_object*self.num_obj_cls, -1)
			zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(hidden))

			# eq(4)
			rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(hidden))

			#eq(5)
			hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * hidden))

			hidden = (1 - zv) * hidden + zv * hv
			hidden = hidden.view(num_object, self.num_obj_cls, -1)


		output = torch.cat((hidden.view(num_object*self.num_obj_cls, -1), 
							input_ggnn.repeat(1, self.num_obj_cls).view(num_object*self.num_obj_cls, -1)), 1)
		output = self.fc_output(output)
		output = self.ReLU(output)
		obj_dists = self.fc_obj_cls(output.view(-1, self.num_obj_cls * self.output_dim))
		return obj_dists



class GGNNRel(nn.Module):
	def __init__(self, num_rel_cls=51, time_step_num=3, hidden_dim=512, output_dim=512, use_knowledge=True, prior_matrix=''):
		super(GGNNRel, self).__init__()
		self.num_rel_cls = num_rel_cls
		self.time_step_num = time_step_num
		self.matrix = np.load(prior_matrix).astype(np.float32)
		self.use_knowledge = use_knowledge

		self.fc_eq3_w = nn.Linear(2 * hidden_dim, hidden_dim)
		self.fc_eq3_u = nn.Linear(hidden_dim, hidden_dim)
		self.fc_eq4_w = nn.Linear(2 * hidden_dim, hidden_dim)
		self.fc_eq4_u = nn.Linear(hidden_dim, hidden_dim)
		self.fc_eq5_w = nn.Linear(2 * hidden_dim, hidden_dim)
		self.fc_eq5_u = nn.Linear(hidden_dim, hidden_dim)

		self.fc_output = nn.Linear(2 * hidden_dim, output_dim)
		self.ReLU = nn.ReLU(True)
		self.fc_rel_cls = nn.Linear((self.num_rel_cls + 2) * output_dim, self.num_rel_cls)

	def forward(self, rel_inds, sub_obj_preds, input_ggnn):
		(input_rel_num, node_num, _) = input_ggnn.size()
		assert input_rel_num == len(rel_inds)
		batch_in_matrix_sub = np.zeros((input_rel_num, 2, self.num_rel_cls), dtype=np.float32)
		
		if self.use_knowledge: # construct adjacency matrix depending on the predicted labels of subject and object.
			for index, rel in enumerate(rel_inds):
				batch_in_matrix_sub[index][0] = \
					self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
				batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
		else:
			for index, rel in enumerate(rel_inds):
				batch_in_matrix_sub[index][0] = 1.0 / float(self.num_rel_cls)
				batch_in_matrix_sub[index][1] = batch_in_matrix_sub[index][0]
		batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False).cuda()
		del batch_in_matrix_sub

		hidden = input_ggnn
		for t in range(self.time_step_num):
			# eq(2)
			# becase in this case, A^(out) == A^(in), so we use function "repeat"
			# What is A^(out) and A^(in)? Please refer to paper "Gated graph sequence neural networks"
			av = torch.cat((torch.bmm(batch_in_matrix_sub_gpu, hidden[:, 2:]), 
							torch.bmm(batch_in_matrix_sub_gpu.transpose(1, 2), hidden[:, :2])), 1).repeat(1, 1, 2)
			av = av.view(input_rel_num * node_num, -1)
			flatten_hidden = hidden.view(input_rel_num * node_num, -1)
			# eq(3)
			zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_hidden))
			# eq(4)
			rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq3_u(flatten_hidden))
			#eq(5)
			hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_hidden))
			flatten_hidden = (1 - zv) * flatten_hidden + zv * hv
			hidden = flatten_hidden.view(input_rel_num, node_num, -1)

		output = torch.cat((flatten_hidden, input_ggnn.view(input_rel_num * node_num, -1)), 1)
		output = self.fc_output(output)
		output = self.ReLU(output)

		rel_dists = self.fc_rel_cls(output.view(input_rel_num, -1))
		return rel_dists


class ResBlock(nn.Module):
	def __init__(self, input_channel, output_channel, stride_size, bias=True):
		super(ResBlock, self).__init__()
		self.input_channel = input_channel
		self.output_channel = output_channel
		self.stride_size = stride_size
		self.k = 5

		self.conv = nn.Sequential(nn.Conv2d(2*input_channel, output_channel, 1, padding=0, bias=False), \
								  nn.BatchNorm2d(output_channel, affine=True), nn.ReLU())

	def knn(self, x, k):
		inner = -2*torch.matmul(x.transpose(2, 1), x)
		xx = torch.sum(x**2, dim=1, keepdim=True)
		pairwise_distance = -xx - inner - xx.transpose(2, 1)

		idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
		return idx

	def get_graph_feature(self, x, k=5, idx=None):
		batch_size = x.size(0)
		num_points = x.size(2)
		x = x.view(batch_size, -1, num_points)
		if idx is None:
			idx = self.knn(x, k=k)	 # (batch_size, num_points, k)
		#device = torch.device('cuda')

		#idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
		idx_base = torch.arange(0, batch_size).view(-1, 1, 1)*num_points
		
		idx_base = idx_base.cuda().long()
		idx_base = Variable(idx_base)               

		idx = idx + idx_base

		idx = idx.view(-1)

		_, num_dims, _ = x.size()

		x = x.transpose(2, 1).contiguous()	 # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #	batch_size * num_points * k + range(0, batch_size*num_points)
		feature = x.view(batch_size*num_points, -1)[idx, :]
		feature = feature.view(batch_size, num_points, k, num_dims) 
		x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
		
		feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

		return feature
	def forward(self, input_x):
		x = self.get_graph_feature(input_x)
		x = self.conv(x)
		x = x.max(dim = -1, keepdim = False)[0]
		return x + input_x


class ResGCN(nn.Module):
	def __init__(self, num_rel_cls=51, hidden_dim=512, output_dim=512, k = 5):
		super(ResGCN,self).__init__()
		self.num_rel_cls = num_rel_cls
		self.gcn = nn.Sequential(nn.Conv1d(3, 64, 1, padding=0, bias=False), nn.BatchNorm1d(64, affine=True), nn.ReLU(), \
								 ResBlock(64,64,2),ResBlock(64,64,2), ResBlock(64,64,3), \
								 nn.Conv1d(64,1,1,padding=0, bias=False), nn.ReLU())


		self.matrix = np.load('./prior_matrices/rel_matrix.npy').astype(np.float32)

		self.fc_cls = nn.Linear(output_dim, self.num_rel_cls )	


	def forward(self, rel_inds, sub_obj_preds, input_ggnn):
		(input_rel_num, node_num, _) = input_ggnn.size()
		
		assert input_rel_num == len(rel_inds)

		batch_in_matrix_sub = np.zeros((input_rel_num, self.num_rel_cls), dtype=np.float32)
		for index, rel in enumerate(rel_inds):
			batch_in_matrix_sub[index] = \
				self.matrix[sub_obj_preds[index, 0].cpu().data, sub_obj_preds[index, 1].cpu().data]
			
		batch_in_matrix_sub_gpu = Variable(torch.from_numpy(batch_in_matrix_sub), requires_grad=False).cuda()
		del batch_in_matrix_sub

		gcn = self.gcn(input_ggnn)
		
		rel_dists = self.fc_cls(gcn.view(input_rel_num,-1))
		
		rel_dists = rel_dists + batch_in_matrix_sub_gpu


		return rel_dists

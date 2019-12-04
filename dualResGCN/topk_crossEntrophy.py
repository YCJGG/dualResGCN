import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class topk_crossEntrophy(nn.Module):
    def __init__(self, weight=None, size_average=None,ignore_index=-100, reduce=None, reduction='none'):
        super(topk_crossEntrophy, self).__init__()
        
        self.ignore_index = ignore_index
        
        self.weight = weight
        self.ignore_index = ignore_index

        self.loss = nn.NLLLoss(weight=self.weight,ignore_index=self.ignore_index, reduce=False).cuda()
    def forward(self, input, target, top_k):
        self.top_k = top_k
        loss = self.loss(F.log_softmax(input, dim=1), target)
#        print(loss)
        
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            valid_loss, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))    
            return torch.mean(valid_loss)

#a = torch.randn((10, 2)).cuda()
#a.normal_()
#b = np.random.randint(2, size=10)
#b = torch.from_numpy(b.astype(np.float32)).type(torch.LongTensor)
#b = b.cuda()
#topk_loss = topk_crossEntrophy(top_k=0.7)
#loss = topk_loss(Variable(a, requires_grad=True), Variable(b))
#print(loss.detach().numpy())

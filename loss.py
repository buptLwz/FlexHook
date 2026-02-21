import torch
from torch import nn, log
import torch.nn.functional as F
from torch.autograd import Variable
#from utils import ID_TO_POS_BBOX_NUMS
from info_nce import InfoNCE

__all__ = [
    'SimilarityLoss',
    'multiSimilarityLoss'
]


class SimilarityLoss(nn.Module):
    """

    TODO: reset
    """
    def __init__(
            self,
            rho: float = None,
            gamma: float = 2.,
            reduction: str = 'mean',
    ):
        super().__init__()
        self.rho = rho  # pos/neg samples
        self.gamma = gamma  # easy/hard samples
        self.reduction = reduction

    #def forward(self, scores, labels):
    def forward(self, logits,labels):
        #sim = F.cosine_similarity(outfeats.unsqueeze(0), tgtfeats.unsqueeze(1),dim=-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")#,pos_weight=torch.tensor(2.0,device=sim.device))
        weights = 1
        #print(loss)
        if self.gamma is not None:
            logits = logits.sigmoid()
            p_t = logits * labels + (1 - logits) * (1 - labels)
            weights *= ((1 - p_t) ** self.gamma)#.mean(-1)

        if self.rho is not None:
            weights *= self.rho * labels + (1 - labels)

        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class multiSimilarityLoss(nn.Module):
    """
    Ref:
    TODO: reset
    """
    def __init__(
            self,
            #rho: float = None,
            #gamma: float = 2.,
            #reduction: str = 'sum',
    ):
        super().__init__()
        self.tao = 0.1
        #self.rho = rho  # pos/neg samples
        #self.gamma = gamma  # easy/hard samples
        #self.reduction = reduction
        self.loss_fn = InfoNCE(negative_mode='my')

    #def forward(self, sim_logits,labels):
    def forward(self, outfeats,tgtfeats,labels):

        loss = self.loss_fn(outfeats,tgtfeats,labels,negative_keys=None)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1) #bn,1
        #input BN,C

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()/10
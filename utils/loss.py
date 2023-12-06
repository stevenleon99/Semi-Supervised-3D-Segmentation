import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.ndimage as nd
from matplotlib import pyplot as plt
from torch import Tensor, einsum
from pdb import set_trace as bp


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]

        return dice_loss_avg

class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target, name, TEMPLATE):
        '''
        TEMPLATE: '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        predic:  [2, 32, 96, 96, 96] 
        target:  [2, 32, 96, 96, 96]
        '''
        total_loss = []
        predict = F.sigmoid(predict)
        B = predict.shape[0]
        organ_list = TEMPLATE['01']  
        total_loss = torch.zeros(B, len(organ_list))
        # task_dic = {'colon':'10','hepaticvessel':'08','liver':'03','lung':'06','pancreas':'07','spleen':'09'}
        for b in range(B):
            # dataset_index = int(name[b][0:2])
            # if dataset_index == 10:
            #     if name[17:19].isdigit():
            #         template_key = name[0:2] + '_' + name[17:19]
            #     else:
            #         task_key = name.split('_')[2]
            #         template_key = name[0:2] + '_' + task_dic[task_key]
            # elif dataset_index == 1:
            #     if int(name[b][-2:]) >= 60:
            #         template_key = '01_2'
            #     else:
            #         template_key = '01'
            # else:
            #     template_key = name[b][0:2]
            
            for organ in organ_list:
                total_loss[b,organ-1] = self.dice(predict[b, organ-1], target[b, organ-1])
            
        return total_loss.mean()

        

class Multi_BCELoss(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target, name, TEMPLATE):
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'

        total_loss = []
        B = predict.shape[0]

        for b in range(B):
            # dataset_index = int(name[b][0:2])
            # if dataset_index == 10:
            #     template_key = name[b][0:2] + '_' + name[b][17:19]
            # elif dataset_index == 1:
            #     if int(name[b][-2:]) >= 60:
            #         template_key = '01_2'
            #     else:
            #         template_key = '01'
            # else:
            #     template_key = name[b][0:2]
            organ_list = TEMPLATE['01']
            for organ in organ_list:
                ce_loss = self.criterion(predict[b, organ-1], target[b, organ-1])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)

        # print(name, total_loss, total_loss.sum()/total_loss.shape[0])

        return total_loss.sum()/total_loss.shape[0]

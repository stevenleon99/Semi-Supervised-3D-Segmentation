import torch
import torch.nn as nn
from monai import transforms
import random
import numpy as np
from pdb import set_trace as bp



class SemiLoss(nn.Module):
    def __init__(self):
        super(SemiLoss, self).__init__()
        
    def generate_consistency_pair(self,input,roi,transform_type='random'):
        w,h,d = roi
        if transform_type == 'random':
            transform_type = random.choice(['flip','rotate','offset'])
        input_ = input[:,:,:w,:h,:d]
        
        if transform_type =='flip': # Flip
            spatial_axis = random.choice([0,1,2])
            transform = transforms.Flip(spatial_axis)
            inverse_transform = transform
            input_t = []
            for B in range(input_.shape[0]):
                input_t.append(transform(input_[B,:,:,:,:]))
            input_t = torch.stack(input_t,dim=0)
            
        elif transform_type == 'rotate': # Rotate
            spatial_axis = np.random.choice([0,1,2],size=2,replace=False)
            transform = transforms.Rotate90(k=1,spatial_axes=spatial_axis)
            inverse_transform = transforms.Rotate90(k=-1,spatial_axes=spatial_axis)
            input_t = []
            for B in range(input_.shape[0]):
                input_t.append(transform(input_[B,:,:,:,:]))
            input_t = torch.stack(input_t,dim=0)
        
        elif transform_type == 'offset': # Offset
            spatial_axis = random.choice([2,3,4])
            offset = 5
            if spatial_axis == 2:
                input_t = input[:,:,offset:w+offset,:h,:d]
                inverse_transform = np.array([[offset,-1,0,-1,0,-1],[0,-1-offset,0,-1,0,-1]])
            elif spatial_axis == 3:
                input_t = input[:,:,:w,offset:h+offset,:d]
                inverse_transform = np.array([[0,-1,offset,-1,0,-1],[0,-1,0,-1-offset,0,-1]])
            else:  
                input_t = input[:,:,:w,:h,offset:d+offset]
                inverse_transform = np.array([[0,-1,0,-1,offset,-1],[0,-1,0,-1,0,-1-offset]])
        
        return input_,input_t,inverse_transform
    
    def forward(self, logis,logis_t,transform):
        if type(transform) == np.ndarray:
            logis = logis[:,:,transform[0,0]:transform[0,1],transform[0,2]:transform[0,3],transform[0,4]:transform[0,5]]
            logis_t = logis_t[:,:,transform[1,0]:transform[1,1],transform[1,2]:transform[1,3],transform[1,4]:transform[1,5]]
        else:
            logis_t_ = []
            for B in range(logis.shape[0]):
                logis_t_.append(transform(logis_t[B,:,:,:,:]))
            logis_t = torch.stack(logis_t_,dim=0)

        loss = torch.mean((logis - logis_t)**2)
        
        return loss
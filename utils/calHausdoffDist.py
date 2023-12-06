from monai.metrics import hausdorff_distance
from monai.metrics import surface_distance
import monai.metrics as metrics
import matplotlib.pyplot as plt
import os
from os.path import isfile
import glob
import numpy as np
import nibabel #req numpy ver. <= 1.21.0
import torch
import torch.nn as nn

Label_src_Root = 'data/PublicAbdominalData/01_Multi-Atlas_Labeling/label'
log_dir = 'logs'
# change these two direct to specific pred
Label_pred_src_Root = 'logs/valid_saved_log/01_Multi-Atlas_Labeling/swinunetr_0.3fs_650cp_rot/01_Multi-Atlas_Labeling'
log_name = 'rotate_650cp_hd_dist.txt'
pred_organ = {'1':'spleen', '3':'kidney_left', '4':'gall_bladder', '6':'liver', '7':'stomach', '11':'pancreas', '14':'duodenum'}
'''
    choose specific label {1 'Spleen', 2:'Right Kidney', 3 'Left Kidney',
                           4 'Gall Bladder', 5 'Esophagus',:6 'Liver', 
                           7 'Stomach', 11 'Pancreas', 14 'Duodenum'}
'''
def getPredFileNameList():
    '''
    key = 'img0061', 'img0062' ...
    value = [path to different organs'.nii.gz], [], ...
    '''
    # find image name list
    imgListFileDir = {} 
    imgNameList = []
    for dirpath, dirnames, filenames in os.walk(Label_pred_src_Root):
        if len(filenames) >= len(pred_organ):
            # print(f"Found directory: {dirpath}")
            # print(f'Pred_organ list: {filenames}')
            value = []
            key = dirpath.split('\\')[-1]
            imgNameList.append(key)
            for file in sorted(filenames):
                value.append(os.path.join(dirpath, file))
            imgListFileDir.update({key:value})
    # print(os.path.exists(imgListFileDir['img0061'][0])) # check file exist 
    # print(imgNameList)   
    return imgListFileDir, imgNameList

def findKey(str, dict):
    key = ''
    for k,v in dict.items():
        if v in str:
            key = k
    return key

if __name__ == "__main__":
    print("====== Calculating Hausdoff Distance =====")
    
    organPathList, imgNameList = getPredFileNameList()
    for k,v in organPathList.items():
        fileName_gt = 'label'+k[3:7]+'.nii.gz'
        label_gt = os.path.join(Label_src_Root, fileName_gt)
        label_gt = nibabel.load(label_gt)
        for organ_path in v:
            key = findKey(organ_path, pred_organ)
            if key:
                print(f">> processing with organ: {pred_organ[key]} img: {fileName_gt}")
                # import pred and ground truth
                label_pred = nibabel.load(organ_path)
                label_pred = np.asanyarray(label_pred.dataobj) > 0
                
                label_organ_gt = np.asanyarray(label_gt.dataobj)
                # change to one-hot format and select
                label_organ_gt =  np.asanyarray(label_gt.dataobj) == int(key)
                
                # expand the dimension to match b,c,h,w,d
                label_pred = np.expand_dims(label_pred, axis=0)
                label_pred = np.expand_dims(label_pred, axis=0)
                label_organ_gt = np.expand_dims(label_organ_gt, axis=0)
                label_organ_gt = np.expand_dims(label_organ_gt, axis=0)

                result = hausdorff_distance.compute_hausdorff_distance(label_organ_gt, label_pred, distance_metric="euclidean", percentile=95)
                # write with labelName, organ, hd_dist
                with open(os.path.join(log_dir, log_name), '+a') as f:
                    f.write(f'{fileName_gt}, {pred_organ[key]}, {result.item()} \n')
                     
    # # traverse each slice
    # # for i in range(label_pred.shape[4]):
    # #     result = hausdorff_distance.compute_hausdorff_distance(label_gt[:,:,:,:,i], label_pred[:,:,:,:,i])
    # #     print(f"index: {i}", result)
    
    # # liver:           tensor([[19.8746]], dtype=torch.float64)
    # # Spleen:          tensor([[5.]], dtype=torch.float64)
    # # LeftKidney:      tensor([[13.]], dtype=torch.float64)
    # # Gall Baldder:    tensor([[65.5439]], dtype=torch.float64)
    # # Stomach:         tensor([[107.7822]], dtype=torch.float64)
    # # Pancreas:        tensor([[23.2594]], dtype=torch.float64)
    # # Duodenum:        tensor([[40.6325]], dtype=torch.float64)
    

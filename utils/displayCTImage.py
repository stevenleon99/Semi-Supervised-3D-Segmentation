import matplotlib.pyplot as plt
import os
from os.path import isfile
import glob
import numpy as np
import nibabel #req numpy ver. <= 1.21.0
import math

Filename = '01_Multi-Atlas_Labeling_train.txt'
FileRoot = 'dataset/dataset_list/'
ImgRoot = 'data/PublicAbdominalData/'
LabelRoot = 'data/PublicAbdominalData/'
Img_src_Root = 'data/PublicAbdominalData/01_Multi-Atlas_Labeling/img/img0061.nii.gz'
Label_src_Root = 'data/PublicAbdominalData/01_Multi-Atlas_Labeling/label/label0061.nii.gz'
Label_pred_src_Root = 'logs/valid_saved_log/01_Multi-Atlas_Labeling/img0061_invertd/backbones/swinunetr/segmentations/im0/im0__liver.nii.gz'

def diplayTruthNPred(trip_dir, size=6, figsize=(12, 12)):
    '''
    | img | gt | pred | img | gt | pred |
    | img | gt | pred | img | gt | pred |
    ...
    
    '''
    img = nibabel.load(trip_dir[0])
    label_gt = nibabel.load(trip_dir[1])
    label_pred = nibabel.load(trip_dir[2])
    img =  np.asanyarray(img.dataobj)
    label_gt =  np.asanyarray(label_gt.dataobj) == 6
    '''
    choose specific label {1 'Spleen', 2:'Right Kidney', 3 'Left Kidney',
                           4 'Gall Bladder', 5 'Esophagus',:6 'Liver', 
                           7 'Stomach', 11 'Pancreas', 14 'Duodenum'}
    '''
    label_pred =  np.asanyarray(label_pred.dataobj)
    # print("shape of slice img: ", img.shape)
    # print("shape of slice label: ", label_gt.shape)
    num_slice = img.shape[2]
    slice_interval = math.floor((num_slice/2)/(size**2/2)) # slice interval : int
    slice_start = math.floor(num_slice/2) - slice_interval*size + 25 # start display slice number: int
    
    # display ground truth image grid
    plt.figure(figsize=figsize)
    count_image = 1
    for idx in range(1, size*size+1, 3):
        plt.subplot(size, size, idx)
        slice_image_idx = slice_start + slice_interval * count_image
        plt.imshow(img[:, :, slice_image_idx], cmap="gray")
        plt.title(f'slice number: {slice_image_idx}')
        plt.subplot(size, size, idx+1)
        plt.imshow(label_gt[:, :, slice_image_idx])
        plt.subplot(size, size, idx+2)
        plt.imshow(label_pred[:, :, slice_image_idx])
        count_image += 1
        
    plt.tight_layout()    
    # plt.show()
    plt.savefig("./output/pred_output.jpg")
    
    

def displayGroundTruth(pair_dir, size=4, figsize=(12, 12)): 
    '''
    | img | gt | img | gt |
    | img | gt | img | gt |
    ...
    
    '''
    img = nibabel.load(pair_dir[0])
    label = nibabel.load(pair_dir[1])
    img =  np.asanyarray(img.dataobj)
    label =  np.asanyarray(label.dataobj)
    # print("shape of slice img: ", img.shape)
    # print("shape of slice label: ", label.shape)
    
    num_slice = img.shape[2]
    slice_interval = math.floor((num_slice/2)/(size**2/2)) # slice interval : int
    slice_start = math.floor(num_slice/2) - slice_interval*size + 25 # start display slice number: int
    
    # display ground truth image grid
    plt.figure(figsize=figsize)
    count_image = 1
    for idx in range(1, size*size+1, 2):
        plt.subplot(size, size, idx)
        slice_image_idx = slice_start + slice_interval * count_image
        plt.imshow(img[:, :, slice_image_idx], cmap="gray")
        plt.title(f'slice number: {slice_image_idx}')
        plt.subplot(size, size, idx+1)
        plt.imshow(label[:, :, slice_image_idx])
        count_image += 1
        
    plt.tight_layout()    
    # plt.show()
    plt.savefig("./output/gt_output.jpg")


def getImgNLabelPair(dir, filename):
    pass

if __name__ == "__main__":
    
    # # find image name list
    # imgListFileDir = ''
    # for dirpath, dirnames, filenames in os.walk(FileRoot):
    #     print(f"Found directory: {dirpath}")
    #     for file_name in filenames:
    #         if file_name == Filename:
    #             imgListFileDir = os.path.join(FileRoot, file_name)
    #             print(f"Found target file: {Filename}")
    
    # # construct the img and label pair path
    # pair = []
    # with open(imgListFileDir, 'r') as f:
    #     for line in f:
    #         files = line.split()
    #         assert (len(files) == 2), "Img Label pair should be 2, but got {}".format(len(files))
    #         if isfile(os.path.join(ImgRoot, files[0])) and isfile(os.path.join(LabelRoot, files[1])):
    #             pair.append([os.path.join(ImgRoot, files[0]), 
    #                          os.path.join(LabelRoot, files[1])])
    #         else: print(f"{files[0]} or {files[1]} not existed")
    # print("total img and label pair: ", len(pair))
    
    
    pair = [Img_src_Root, Label_src_Root]
    # display ground truth image
    displayGroundTruth(pair, size = 6)
    triple = [Img_src_Root, Label_src_Root, Label_pred_src_Root]
    # display ground truth and pred image
    diplayTruthNPred(triple, size=6)
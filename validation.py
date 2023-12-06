import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import shutil
import nibabel as nib
import csv
from pdb import set_trace as bp

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import (
    AsDiscrete,
    BatchInverseTransform,
    ) 
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from model.SwinUNETR_partial import SwinUNETR
from model.Universal_model import Universal_model

from dataset.dataloader_validation import get_validation_loader, getDataInverted, saveTransImage

from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key, pseudo_label_all_organ, pseudo_label_single_organ, save_organ_label,calculate_dice
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS,ORGAN_NAME_LOW
from utils.utils import organ_post_process, threshold_organ,create_entropy_map,save_soft_pred,invert_transform
torch.multiprocessing.set_sharing_strategy('file_system')



def validation(model, ValLoader, val_transforms, args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.eval()
    
    dice_avg = {} # dice for different organs
    count = {} # num for different organs
    organ_index_all = TEMPLATE['01']  # '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    for organ_index in organ_index_all:
        '''
        ORGAN_NAME_LOW = ['spleen',   'kidney_right',        'kidney_left',        'gall_bladder',      'esophagus', 
                          'liver',    'stomach',             'aorta',              'postcava',          'portal_vein_and_splenic_vein',
                          'pancreas', 'adrenal_gland_right', 'adrenal_gland_left', 'duodenum', ...etc]
        '''
        organ_name = ORGAN_NAME_LOW[organ_index-1]
        dice_avg.update({organ_name:0})
        count.update({organ_name:0})
        
    
    for index, batch in enumerate(ValLoader):
        print(f"======== test/valid {index+1} / {len(ValLoader)} ========")
        
        #################### establish directory tree ####################
        if args.original_label: # img with label
            image, label, name_lbl, name_img = batch["image"].cuda(), batch["label"], batch["name_lbl"],batch["name_img"]
            print(f">{name_img} shape: {image.shape} >{name_lbl} shape: {label.shape}")
            # saveTransImage(batch, save_dir, args) # # save the transformed .nii
            
            image_file_path = os.path.join(args.data_root_path,name_img[0] +'.nii.gz')
            lbl_file_path = os.path.join(args.data_root_path,name_lbl[0] +'.nii.gz')
            case_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1])
            pseudo_label_save_path = os.path.join(case_save_path,'backbones',args.backbone)
            
            if not os.path.isdir(pseudo_label_save_path):
                os.makedirs(pseudo_label_save_path)
                
            organ_seg_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'segmentations')
            organ_entropy_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'entropy')
            organ_soft_pred_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'soft_pred')
            destination_ct = os.path.join(case_save_path,'ct.nii.gz')
            
            if not os.path.isfile(destination_ct):
                shutil.copy(image_file_path, destination_ct)
                print("Image File copied successfully.") # copy the ori image [*file*].nii.gz
            
            destination_lbl = os.path.join(case_save_path,'original_label.nii.gz')
            if not os.path.isfile(destination_lbl):
                shutil.copy(lbl_file_path, destination_lbl)
                print("Label File copied successfully.") # copy the ori label [*file*].nii.gz
            # affine_temp = nib.load(destination_ct) # (512, 512, 138)
            affine_temp = nib.load(destination_ct).affine # (4, 4) the transformation matrix
                    
        else:
            image,name_img = batch["image"].cuda(),batch["name_img"]
            image_file_path = os.path.join(args.data_root_path,name_img[0] +'.nii.gz')
            case_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1])
            pseudo_label_save_path = os.path.join(case_save_path,'backbones',args.backbone)
            if not os.path.isdir(pseudo_label_save_path):
                os.makedirs(pseudo_label_save_path)
            organ_seg_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'segmentations')
            organ_entropy_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'entropy')
            organ_soft_pred_save_path = os.path.join(save_dir,name_img[0].split('/')[0],name_img[0].split('/')[-1],'backbones',args.backbone,'soft_pred')
            print(image_file_path)
            print(image.shape)
            print(name_img)
            destination_ct = os.path.join(case_save_path,'ct.nii.gz')
            if not os.path.isfile(destination_ct):
                shutil.copy(image_file_path, destination_ct)
                print("Image File copied successfully.")
            affine_temp = nib.load(destination_ct).affine

        #####################################################################
        
        with torch.no_grad():
            # [1, 1, 234, 199, 275] -> [1(batch num), 32(classification), 234(H), 199(W), 275(D)]
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.75, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
            # print(pred_sigmoid.mean(), pred_sigmoid.std()) # metatensor(0.0026, device='cuda:0') metatensor(0.0448, device='cuda:0')
            
        # detect the existence of organs by threshould value
        pred_hard = threshold_organ(pred_sigmoid,args) # [1, 32, 234, 199, 275] - > [1, 32, 234, 199, 275]
        pred_hard = pred_hard.cpu()
        
        torch.cuda.empty_cache()
        
        B = pred_hard.shape[0]
        for b in range(B):
            organ_list_all = TEMPLATE['all'] # post processing all organ
            pred_hard_post,total_anomly_slice_number = organ_post_process(pred_hard.numpy(), organ_list_all,case_save_path,args)
            pred_hard_post = torch.tensor(pred_hard_post)
        
        
        if args.evaluate:
            dice_percase = {}
            organ_index_all = TEMPLATE['01']
            label_onehot = onehot_encoding(label,organ_index_all) #[1, 14, 234, 199, 275]
            for organ_index in organ_index_all:
                pseudo_label_single = pseudo_label_single_organ(pred_hard_post,organ_index,args) # pred single organ label  [1, 1, 234, 199, 275]
                organ_name = ORGAN_NAME_LOW[organ_index-1]
                dice_template,count_template = calculate_dice(pseudo_label_single.squeeze(0),label_onehot[:,organ_index-1])
                dice_percase.update({organ_name:dice_template})
                dice_avg[organ_name] += dice_template
                count[organ_name] += count_template
            # print('dice_percase: ',dice_percase)
            if write_log_mode == 1:
                with open(log_file, "+a") as f:
                    f.write(f"{name_img[0].split('/')[-1]}, ")
                    for cls, value in dice_percase.items():
                        f.write(f'{cls}, {value}, ')
                    f.write('\n')
    
        if args.store_result:
            if not os.path.isdir(organ_seg_save_path):
                os.makedirs(organ_seg_save_path)
            organ_index_all = TEMPLATE['01']
            for organ_index in organ_index_all:
                pseudo_label_single = pseudo_label_single_organ(pred_hard_post,organ_index,args)
                # print("Shape before invert: ", pseudo_label_single.shape)
                organ_name = ORGAN_NAME_LOW[organ_index-1]
                pseuLbl = pseudo_label_single.cpu() # dict_keys(['image', 'label', 'name_lbl', 'name_img', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord', 'spleen'])
                organ_invertd = getDataInverted(organ_name, pseuLbl, organ_seg_save_path, batch["name_img"][0], args) # invert the pseudo label with external invertd
                # batch[organ_name]=pseuLbl
                # BATCH = invert_transform(organ_name, batch, val_transforms)
                # organ_invertd = np.squeeze(BATCH[0][organ_name].numpy(),axis = 0)
                # organ_save = nib.Nifti1Image(organ_invertd,np.eye(4))
                # new_name = os.path.join(organ_seg_save_path, organ_name+'.nii.gz')
                # print('organ seg saved in path: %s'%(new_name))
                # nib.save(organ_save,new_name)
                
        # if args.store_entropy:
        #     organ_index_target = TEMPLATE['target']
        #     if not os.path.isdir(organ_entropy_save_path):
        #         os.makedirs(organ_entropy_save_path)
        #     for organ_idx in organ_index_target:
        #         organ_entropy = create_entropy_map(pred_sigmoid,organ_idx)
        #         organ_name_target = ORGAN_NAME_LOW[organ_idx-1]
        #         batch[organ_name_target] = organ_entropy.cpu()
        #         BATCH = invert_transform(organ_name_target,batch,val_transforms)
        #         organ_invertd = np.squeeze(BATCH[0][organ_name_target].numpy(),axis = 0)*255
        #         organ_save = nib.Nifti1Image(organ_invertd.astype(np.uint8),affine_temp)
        #         new_name = os.path.join(organ_entropy_save_path, organ_name_target+'.nii.gz')
        #         print('organ entropy saved in path: %s'%(new_name))
        #         nib.save(organ_save,new_name)

        # if args.store_soft_pred:
        #     organ_index_target = TEMPLATE['all']
        #     if not os.path.isdir(organ_soft_pred_save_path):
        #         os.makedirs(organ_soft_pred_save_path)
        #     for organ_idx in organ_index_target:
        #         organ_pred_soft_save = save_soft_pred(pred_sigmoid,pred_hard_post,organ_idx,args)
        #         organ_name_target = ORGAN_NAME_LOW[organ_idx-1]
        #         batch[organ_name_target] = organ_pred_soft_save.cpu()
        #         BATCH = invert_transform(organ_name_target,batch,val_transforms)
        #         organ_invertd = np.squeeze(BATCH[0][organ_name_target].numpy(),axis= 0)*255
        #         organ_save = nib.Nifti1Image(organ_invertd.astype(np.uint8),affine_temp)
        #         new_name = os.path.join(save_dir, organ_soft_pred_save_path, organ_name_target+'.nii.gz')
        #         print('organ soft pred saved in path: %s'%(new_name))
        #         nib.save(organ_save,new_name)
     
    #     torch.cuda.empty_cache()
        
    # if args.evaluate:
    #     for k,v in dice_avg.items():
    #         if count[k] != 0:
    #             dice_avg[k] = v/count[k]
    #         else:
    #             dice_avg[k] = -1
    #     print('dice_avg: ',dice_avg)
    #     dice_avg_avg = []
        
    #     for dice in dice_avg.values():
    #         if dice != -1:
    #             dice_avg_avg.append(dice)
    #     dice_avg_avg = torch.stack(dice_avg_avg)        
    #     dice_avg_avg = torch.mean(dice_avg_avg)
    #     print('dice_avg_avg: ',dice_avg_avg)


def onehot_encoding(label, classes):
    num_classes = len(classes)
    label_onehot = torch.zeros(label.size(0), num_classes, label.size(2),label.size(3), label.size(4)).cuda()
    for B in range(label.size(0)):
        for i in classes:
            label_onehot[B,i-1,:,:,:] = (label[B]==i)

    return label_onehot


write_log_mode = 1
if write_log_mode == 1:
    log_file = 'logs/accuracy_info.txt'
write_arg = 0

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--save_dir', default='Nvidia/old_fold0', help='The dataset save path')
    ## model load
    parser.add_argument('--resume', default='None', help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='None', help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--original_label',action="store_true", default=False,help='whether dataset has original label')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--store_entropy', action="store_true", default=False, help='whether save entropy map')
    parser.add_argument('--store_soft_pred', action="store_true", default=False, help='whether save soft prediction')
    parser.add_argument('--evaluate', action="store_true", default=False, help='whether save soft prediction')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--cpu',action="store_true", default=False, help='whether use cpu')
    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')
    parser.add_argument('--create_dataset',action="store_true", default=False, help='whether create atlas8k')
    
    args = parser.parse_args()

    ##############print the input argument##############
    if write_arg == 1:
        for arg_name, arg_value in vars(args).items():
            print(f'Argument: {arg_name}, Value: {arg_value}')
    
    #################### load model #####################
    if args.backbone == 'swinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    feature_size=24,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    use_checkpoint=False,
                    encoding='word_embedding'
                    )
    else: 
        model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                        in_channels=1,
                        out_channels=NUM_CLASS,
                        backbone=args.backbone,
                        encoding='word_embedding'
                        )
        
    ############# test all epoch check points ###############
    startEpoch = 1700
    interval = 50
    numEpoch = 6
    for i in range(1, numEpoch, 1):
        args.resume = f'saved_check_point/swinunetr_0.3fs_rot/20231121/epoch_{startEpoch + i * interval}.pth'
         
        ############# Load pre-trained weights ###############
        if args.pretrain != 'None':
                print("pretrain weights >>>>>", args.pretrain)
                store_dict = model.state_dict()
                try:
                    model_dict = torch.load(args.pretrain)["state_dict"]
                except:
                    model_dict = torch.load(args.pretrain)["net"]
                    model_dict_ = {}
                    for k,v in model_dict.items():
                        model_dict_[k.replace('module.','')] = v
                    model_dict = model_dict_
                for key in model_dict.keys():
                    if 'out' not in key:
                        store_dict[key] = model_dict[key]
                model.load_state_dict(store_dict) # seems the model_dict (pretrained) are not loaded ??
                print('Use pretrained weights')
        
        if args.resume != 'None':
            print("check point pretrain weights >>>>>", args.resume)
            store_dict = model.state_dict()
            model_dict = torch.load(args.resume)["net"]

            for k,v in model_dict.items():
                if 'out' not in k:
                    k = k.replace('module.', '') if 'module' in k else k
                    store_dict[k] = v

            model.load_state_dict(store_dict)
            print('Use check point pretrained weights')
        #########################################################
        
        model.cuda()

        torch.backends.cudnn.benchmark = True
        
        test_loader, val_transforms = get_validation_loader(args)
        
        validation(model, test_loader, val_transforms, args)


if __name__ == "__main__":
    main()
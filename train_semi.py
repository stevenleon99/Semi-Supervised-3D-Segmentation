import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from torchinfo import summary
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete,SaveImage
from monai.metrics import DiceMetric
import nibabel as nib

from model.SwinUNETR_partial import SwinUNETR
from dataset.dataloader import get_loader_semi
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS,PSEUDO_LABEL_ALL
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.regularization import SemiLoss
from pdb import set_trace as bp
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')


def train_iter_labeled(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE,step,loss_bce_ave,loss_dice_ave):
    batch = next(train_loader)
    x, lbl, name = batch["image"].to(args.device), batch["label"].float(), batch['name']
    B, C, W, H, D = lbl.shape
    y = torch.zeros(B,NUM_CLASS,W,H,D)
    for b in range(B):
        for src,tgt in enumerate(TEMPLATE['01']):
            y[b][src][lbl[b][0]==tgt] = 1
    y = y.to(args.device)
    logit_map = model(x)
    term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
    term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
    loss = term_seg_BCE + term_seg_Dice
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if args.local_rank == 0:
        tqdm.write(
                "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                    args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
            )
    loss_bce_ave += term_seg_BCE.item()
    loss_dice_ave += term_seg_Dice.item()

    return loss_bce_ave, loss_dice_ave     
    
def transform_(x, transform,x_ori=None):
    x_ = []
    if type(transform) == np.ndarray:
        x_ori = x_ori[:,:,transform[0,0]:transform[0,1],transform[0,2]:transform[0,3],transform[0,4]:transform[0,5]]
        x = x[:,:,transform[1,0]:transform[1,1],transform[1,2]:transform[1,3],transform[1,4]:transform[1,5]]
        return x, x_ori
    else:
        for B in range(x.shape[0]):
            x_.append(transform(x[B]))
        x_ = torch.stack(x_)
        return x_

def train_iter_unlabeled(args, train_loader_unlabeled, model, optimizer,loss_consistency,step,loss_consis_avg):
    batch = next(train_loader_unlabeled)
    x,name = batch["image"].to(args.device), batch['name']
    x, x_t, transform = loss_consistency.generate_consistency_pair(x, (args.roi_x, args.roi_y, args.roi_z))
    with torch.no_grad():
        logit_map = model(x)
    logit_map_t = model(x_t)
    term_consis = loss_consistency.forward(logit_map, logit_map_t, transform)
    loss = term_consis * args.alpha
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if args.local_rank == 0:
        tqdm.write(
                "Epoch=%d: Training (%d / %d Steps) (consistency_loss=%2.5f)" % (
                    args.epoch, step, len(train_loader_unlabeled), term_consis.item())
            )
    loss_consis_avg += term_consis.item()
    
    return  loss_consis_avg


def train(args, train_loader, train_loader_unlabeled, model, optimizer, loss_seg_DICE, loss_seg_CE,loss_consistency):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    loss_consistency_ave = 0
    iter_count = 0
    train_loader = iter(train_loader)
    train_loader_unlabeled = iter(train_loader_unlabeled)

    # for step, batch in enumerate(epoch_iterator):
    for step in range(len(train_loader)):
        loss_bce_ave, loss_dice_ave = train_iter_labeled(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE,step,loss_bce_ave,loss_dice_ave)
        if args.train_labeleddata:
            loss_consistency_ave = 0.0
        else:
            loss_consistency_ave = train_iter_unlabeled(args, train_loader_unlabeled, model, optimizer,loss_consistency,step,loss_consistency_ave)
        iter_count += 1
        torch.cuda.empty_cache()

    if args.local_rank == 0:
        train_info = 'Epoch, %d, ave_dice_loss, %2.5f, ave_bce_loss, %2.5f, ave_consist_loss, %2.5f' % (\
                args.epoch, \
                loss_dice_ave/len(train_loader), \
                loss_bce_ave/len(train_loader), \
                loss_consistency_ave/iter_count)
        print(train_info)
        if write_log_mode == 1:
            with open(log_file, 'a') as lf:
                lf.write(str(datetime.now()) + ", " + train_info + '\n')
                

    return loss_dice_ave/len(train_loader), loss_bce_ave/len(train_loader), loss_consistency_ave/iter_count




mod_mode = 1 # flag to show use modification section or not
write_log_mode = 1
write_arg = 1

if write_log_mode == 1: # log loss information
        log_file = f'logs/loss_info.txt'


def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    print(args.device)

    # prepare the 3D model
    model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    feature_size=24,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    use_checkpoint=False,
                    encoding=args.trans_encoding
                    )
    # summary(model, input_size=(1, 1, 96, 96, 96))
    #######################################################
    if mod_mode != 1:
        #Load pre-trained weights
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)["state_dict"]
        for key in model_dict.keys():
            if 'out' not in key:
                store_dict[key] = model_dict[key]

        model.load_state_dict(store_dict)
        print('Use pretrained weights')
    else:
        if args.pretrain != 'none':
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
    ####################################################### 

    if args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding) # torch.Size([32, 512])
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()

    ################# Parallel training(not used) #######################
    if mod_mode != 1:
        if args.dist:
            model = DistributedDataParallel(model, device_ids=[args.device])
    ###################################################################


    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    loss_consistency = SemiLoss().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    
    # for fewshot in args.few_shot:
    
    args.epoch = 0

    ##################### resume from checkpoint #######################
    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch'] + 1
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        torch.backends.cudnn.benchmark = True
        
        print('success resume from ', args.resume)
    ###################################################################


    ########################## dataloader #############################
    train_loader, train_sampler, train_loader_unlabeled, train_sampler_unlabeled = get_loader_semi(args)
    ###################################################################

    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.save_root,args.log_name,str(args.random_seed)))
        print('Writing Tensorboard logs to ', os.path.join(args.save_root,args.log_name,str(args.random_seed)))

    ############################# train ###############################
    while args.epoch <= args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
            train_sampler_unlabeled.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce, loss_consistenct = train(args, train_loader, train_loader_unlabeled, model, optimizer, loss_seg_DICE, loss_seg_CE,loss_consistency)
        
        if rank == 0:
            writer.add_scalar(f'train_dice_loss_fewshot:{args.fewshot}', loss_dice, args.epoch)
            writer.add_scalar(f'train_bce_loss_fewshot:{args.fewshot}', loss_bce, args.epoch)
            writer.add_scalar(f'train_consistency_loss_fewshot:{args.fewshot}', loss_consistenct, args.epoch)
            writer.add_scalar(f'lr_fewshot:{args.fewshot}', scheduler.get_lr(), args.epoch)
    ###################################################################

    
    ###################### Save check point ############################
        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir(os.path.join(args.save_root,args.log_name,str(args.random_seed))):
                os.mkdir(os.path.join(args.save_root,args.log_name,str(args.random_seed)))
            torch.save(checkpoint, os.path.join(args.save_root,args.log_name,str(args.random_seed),'epoch_'+str(args.epoch)+'.pth'))
            print('save model success')
    ###################################################################

        args.epoch += 1

    if args.dist:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='PAOT_v2', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=20, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=5, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    ### for cross_validation 'cross_validation/PAOT_0' 1 2 3 4
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=3, type=int, help='workers numebr for DataLoader')
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
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    # parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
    #                                         '07', '08', '09', '12', '13', '10_03', 
    #                                         '10_06', '10_07', '10_08', '10_09', '10_10'],
    #                                         help='the content for ')
    parser.add_argument('--datasetkey', nargs='+', default=['01'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--gpu', type=str, default='2,3,4')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--fewshot',type=float, default= 0.1,help='whether use few shot')
    parser.add_argument('--save_root',type=str, default='saved_check_point' ,help='few shot num')
    parser.add_argument('--alpha',type=float, default=0.1 ,help='consistency term weight')
    parser.add_argument('--transform_type',type=str,default='random',help='consistency transform')
    parser.add_argument('--train_labeleddata',type=bool, default=False, help='only train the labeled data')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.random_seed)

    #############print the input argument##############
    if write_arg == 1:
        for arg_name, arg_value in vars(args).items():
            print(f'Argument: {arg_name}, Value: {arg_value}')
    

    process(args=args)

if __name__ == "__main__":
    main()

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --uniform_sample

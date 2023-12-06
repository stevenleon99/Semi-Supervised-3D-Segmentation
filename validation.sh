datapath=data/PublicAbdominalData
dataname_valid=01_Multi-Atlas_Labeling_valid
dataname=01_Multi-Atlas_Labeling_test
savepath=logs/valid_saved_log
check_point_pth=saved_check_point/swinunetr_0.3fs_baseline/2345/epoch_750.pth
pretrain_pth=pretrained_weights/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt
dist=1234
bb=swinunetr
bz=1
phase=test # only accept test now


CUDA_VISIBLE_DEVICES=4 python -W ignore validation.py \
--dist True \
--original_label \
--resume $check_point_pth \
--evaluate \
--backbone $bb \
--save_dir $savepath \
--dataset_list $dataname \
--data_root_path $datapath \
--phase $phase \
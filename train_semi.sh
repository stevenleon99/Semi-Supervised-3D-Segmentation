datapath=data/PublicAbdominalData
dataname=01_Multi-Atlas_Labeling_train #file to retrieve train dataname list
savepath=data/outs
dist=2345
bb='swinunetr'
bz=1
fewshot=0.3
type='rotate'
store_num=50
pretrain_weight=pretrained_weights/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt
checkpoint_path=saved_check_point/swinunetr.batchsize_1.wooffset/2345/epoch_650.pth
train_labeleddate=True

python -W ignore -m torch.distributed.launch --nproc_per_node=3 train_semi.py \
--dist True \
--data_root_path $datapath \
--uniform_sample \
--dataset_list $dataname \
--log_name $bb.batchsize_$bz.wooffset \
--backbone $bb \
--store_num 100 \
--lr 1e-4 \
--warmup_epoch 20 \
--fewshot $fewshot \
--batch_size $bz \
--max_epoch 2000 \
--num_workers 0 \
--random_seed $dist \
--transform_type $type \
--store_num $store_num \
--train_labeleddata $train_labeleddate \
--pretrain $pretrain_weight #s> $savepath/$bb.batchsize_$bz.fewshot_$fewshot.seed_$dist.out

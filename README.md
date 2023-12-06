## get pretrained weight
 ```
 cd pretrained_weights
 wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.small_5000ep_f24_lr2e-4_pretrained.pt
 ```

## get BTCV
 ```
cd data\PublicAbdominalData
wget https://www.dropbox.com/s/jnv74utwh99ikus/01_Multi-Atlas_Labeling.tar.gz # 01 Multi-Atlas_Labeling.tar.gz (1.53 GB)
 ```

## training
To reproduce training locally, need to change the env to your local computer

## training result
### to reproduce the result, need to obtain:
 ```
saved_check_point
logs/valid_saved_log/01_Multi-Atlas_Labeling/[predict CT segmentation exams]
 ```
![training result](document/image.png)

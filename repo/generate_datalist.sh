datapath=data/PublicAbdominalData
dataname=01_Multi-Atlas_Labeling
savepath=data/outs
python -W ignore generate_datalist.py --data_path $datapath --dataset_name $dataname --out ./dataset/dataset_list --save_file $dataname.txt
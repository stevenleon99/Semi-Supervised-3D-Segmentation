from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
    RandZoomd,
    RandCropByLabelClassesd,
    RandRotated,
    InvertibleTransform,
    EnsureTyped,
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    Invertd,
    SaveImaged,
    ScaleIntensityd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import os

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import (
    DataLoader, 
    Dataset, 
    list_data_collate, 
    DistributedSampler, 
    CacheDataset, 
    create_test_image_3d, 
    create_test_image_2d, 
    decollate_batch)
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix

import nibabel as nib
from glob import glob

DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

def get_validation_loader(args):
    
    if args.original_label:
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(args.space_x, args.space_y, args.space_z), # 1.5 1.5 1.5
                    mode=('bilinear', 'nearest'),
                ), # process h5 to here
                CropForegroundd(keys=["image", "label"], source_key="image"), # box bounding crop the image
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True,
                ),
                ToTensord(keys=["image", "label"]),
                
            ]
        )
    
    else:
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                AddChanneld(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                # ToTemplatelabeld(keys=['label']),
                # RL_Splitd(keys=['label']),
                Spacingd(
                    keys=["image"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=('bilinear', 'nearest'),
                ), # process h5 to here
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
    )
    
    ############################# check invertibility #############################    
    # keys = ["image", "label"]
    # t = RandRotated(keys)
    # print("RandRotated is invertible: ", isinstance(t, InvertibleTransform))                        # True
    # t = Orientationd(keys, axcodes="RAS") 
    # print("Orientationd is invertible: ", isinstance(t, InvertibleTransform))                       # True
    # t = Spacingd(keys, pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear"),) 
    # print("Spacingd is invertible: ", isinstance(t, InvertibleTransform))                           # True
    # t = ScaleIntensityRanged(keys=keys, 
    #                          a_min=args.a_min, 
    #                          a_max=args.a_max, 
    #                          b_min=args.b_min, b_max=args.b_max, clip=True)
    # print("ScaleIntensityRanged is invertible: ", isinstance(t, InvertibleTransform))               # False
    # t = CropForegroundd(keys=keys, source_key="image")
    # print("CropForegroundd is invertible: ", isinstance(t, InvertibleTransform))                    # True
    
    
    ## test dict part
    if args.original_label:
        test_img = []
        test_lbl = []
        test_name_lbl = []
        test_name_img=[]
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,item + '.txt')):
                name_lbl = line.strip().split()[1].split('.')[0]
                name_img = line.strip().split()[0].split('.')[0]
                test_img.append(os.path.join(args.data_root_path,line.strip().split()[0]))
                test_lbl.append(os.path.join(args.data_root_path,line.strip().split()[1]))
                test_name_lbl.append(name_lbl)
                test_name_img.append(name_img)
        data_dicts_test = [{'image': image, 'label': label, 'name_lbl': name_lbl,'name_img':name_img}
                    for image, label, name_lbl,name_img in zip(test_img, test_lbl, test_name_lbl,test_name_img)]
        print(data_dicts_test)
        # print('test len {}'.format(len(data_dicts_test)))
    else:
        test_img = []
        test_name_img=[]
        for item in args.dataset_list:
            for line in open(os.path.join(args.data_txt_path,item + '.txt')):
                name_img = line.strip().split()[0].split('.')[0]
                test_img.append(os.path.join(args.data_root_path,line.strip().split()[0]))
                test_name_img.append(name_img)
        data_dicts_test = [{'image': image,'name_img':name_img}
                    for image, name_img in zip(test_img, test_name_img)]
        # print('test len {}'.format(len(data_dicts_test)))
    
    if args.phase == 'test':
        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
        return test_loader, val_transforms


def getDataInverted(organ_name, pred_tensor, organ_seg_save_path, args):
    '''
    params:
    ori_img directory of orginal image (e.g. "data/PublicAbdominalData/01_Multi-Atlas_Labeling/img/img0061.nii.gz")
    pred_img tensor of pred
    '''
          
    tempdir = 'temp'
    ############################# fabricate .nii data #############################   
    # for i in range(2):
    #     im, _ = create_test_image_3d(512, 512, 138, num_seg_classes=1) # (512, 512, 138)
    #     n = nib.Nifti1Image(im, np.eye(4))
    #     nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))
    images = sorted(glob(os.path.join(tempdir, "im*.nii.gz"))) # (512,512,138)
    files = [{"img": img} for img in images]
    print(f">Current processing file: {files}, with Organ: {organ_name}")
    
    # define pre transforms
    pre_transforms = Compose(
            [
                LoadImaged(keys=["img"]),
                AddChanneld(keys=["img"]),
                Orientationd(keys=["img"], axcodes="RAS"),
                Spacingd(
                    keys=["img"],
                    pixdim=(args.space_x, args.space_y, args.space_z),
                    mode=('bilinear'),
                ), # process h5 to here
                CropForegroundd(keys=["img"], source_key="img"),
                ScaleIntensityRanged(
                    keys=["img"],
                    a_min=args.a_min,
                    a_max=args.a_max,
                    b_min=args.b_min,
                    b_max=args.b_max,
                    clip=True
                ),
            ]
    )
    
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    
    # define post transforms
    post_transforms = Compose(
        [
            Activationsd(keys=organ_name, sigmoid=False), # already perform Sigmoid for pred
            Invertd(
                keys=organ_name,
                transform=pre_transforms,
                orig_keys="img",
                nearest_interp=True,
                to_tensor=True,
            ),
            AsDiscreted(keys=organ_name, threshold=0.5),
            SaveImaged(keys=organ_name, output_dir=organ_seg_save_path, output_postfix=f"_{str(organ_name)}", resample=False),
        ]
    )

    for _, d in enumerate(dataloader):
        print(f"tr_img_{organ_name}: ", d['img'].shape)  # [1, 1, 234, 187, 275]
        d[organ_name] = pred_tensor
    
    d = [post_transforms(i) for i in decollate_batch(d)]
    print(f"inv_tr_img_{organ_name}: ", d[0][organ_name].shape, " mean: ", d[0][organ_name].mean()) # [1, 512, 512, 138]
    
    ############################# real .nii data invertd test #############################
    # ori_img = "data/PublicAbdominalData/01_Multi-Atlas_Labeling/img/img0061.nii.gz"
    # # ori_img_name = "img0061.nii.gz"
    # data_dicts_test = [{'image': ori_img}]
    # pre_transforms = Compose(
    #     [
    #         LoadImaged(keys=["image"]),
    #         AddChanneld(keys=["image"]),
    #         Orientationd(keys=["image"], axcodes="RAS"),
    #         Spacingd(
    #             keys=["image"],
    #             pixdim=(1.5, 1.5, 1.5),
    #             mode=("bilinear"),
    #         ), # process h5 to here
    #         ScaleIntensityRanged(
    #             keys=["image"],
    #             a_min= -175,
    #             a_max=  250,
    #             b_min=  0,
    #             b_max=  1,
    #             clip=True
    #         ),
    #         CropForegroundd(keys=["image"], source_key="image"),
    #         ToTensord(keys=["image"]),
    #     ]
    # )
    
    # post_transforms = Compose(
    #     [
    #         Activationsd(keys="pred", sigmoid=True),
    #         Invertd(
    #             keys="pred",  # invert the `pred` data field, also support multiple fields
    #             transform=pre_transforms,
    #             orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
    #             # then invert `pred` based on this information. we can use same info
    #             # for multiple fields, also support different orig_keys for different fields
    #             nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
    #             # to ensure a smooth output, then execute `AsDiscreted` transform
    #             to_tensor=True,  # convert to PyTorch Tensor after inverting
    #         ),
    #         AsDiscreted(keys="pred", threshold=0.5),
    #         # SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
    #     ]
    # )
    # dataset = Dataset(data=data_dicts_test, transform=pre_transforms)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # for _, batch in enumerate(dataloader):
    #     img = batch['image']
    #     print(img.shape) # torch.Size([1, 1, 234, 199, 275])
    #     batch['pred'] = torch.randn([1, 1, 512, 411, 138])
    #     print(batch['pred'].shape) # torch.Size([1, 1, 234, 199, 275])
    #     d = [post_transforms(i) for i in decollate_batch(batch)]
    #     print(d[0]['pred'].shape) # torch.Size([1, 1, 234, 199, 275])
    
    return d[0][organ_name]
    
    
def saveTransImage(batch, save_dir, args):
    # # save the transformed .nii
    tr_img = batch["image"].cpu().squeeze(0).squeeze(0).numpy()
    tr_img_save = nib.Nifti1Image(tr_img,np.eye(4))
    file_name = f"trans_{batch['name_img'][0].split('/')[-1]}.nii.gz"
    case_save_path = os.path.join(save_dir,batch["name_img"][0].split('/')[0],batch["name_img"][0].split('/')[-1])
    tr_img_save_dir = os.path.join(case_save_path,file_name)
    nib.save(tr_img_save, tr_img_save_dir)
    

if __name__ == "__main__":
    getDataInverted()
    
import json
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
from os import path
from glob import glob
from tqdm import tqdm

import multiprocessing
from PIL import Image
import numpy as np

import torchvision.transforms as T
import os
#from misc import nested_tensor_from_tensor_list,nested_tensor_from_videos_list
import pickle
import random
merge_set=[['seq_1','seq_9'],['seq_2','seq_10'],['seq_3','seq_11'],['seq_4','seq_12'],['seq_5','seq_13'],['seq_6','seq_14'],['seq_7','seq_15'],['seq_8','seq_16']]


class ISICDataset1(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,data_choice=0,transform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
        #self.videos_dir = path.join(dataset_path,subset_type, 'JPEGImages')
        self.videos_dir = '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/JPEGImages'
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        
        #self.mask_annotations_dir = path.join(dataset_path,subset_type, 'Annotations') 
        self.mask_annotations_dir= '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/Annotations'
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/text.json'

        
        
        with open(metadata_file_path, 'r') as f:
            samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
        

        
        merge_set=[['seq_1','seq_9'],['seq_2','seq_10'],['seq_4','seq_12'],['seq_6','seq_14'],['seq_7','seq_15'],['seq_8','seq_16'],['seq_5','seq_13'],['seq_3','seq_11']]
        
        print(merge_set[data_choice])
        print(merge_set[7-data_choice])
        train_dir=[['seq_1','seq_9'],['seq_2','seq_10'],['seq_4','seq_12'],['seq_6','seq_14'],['seq_7','seq_15'],['seq_8','seq_16'],['seq_5','seq_13'],['seq_3','seq_11']]
        
        
        
        self.train_dir=[]
        for i in train_dir:
            self.train_dir+=i 
        self.valid_dir=self.train_dir

        if self.subset_type=='train':
            data_dir=self.train_dir
        else:
            data_dir=self.valid_dir
        sample_new=[]
        print("train_dir:",self.train_dir)
        print("valid_dir",self.valid_dir)
        for i in range(len(samples_list)):
            if samples_list[i][0] in data_dir:
                sample_new.append(samples_list[i])
              
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = []
        transforms.append(T.Resize(size=(img_size, img_size)))
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)
        with open('/media/cygzz/data/guanjl/Code_self/text.pkl', 'rb') as f:
            all_info_80 = pickle.load(f)  
        self.f=all_info_80
        text=[]
        text_feature=[]
        for i in range(len(self.f)):
            item=self.f[i]
            text.append(item[0])
            text_feature.append(item[1])
        self.text=text
        self.text_feature=text_feature 
        pp=[]
        #self.samples_list=samples_list
        
        self.samples_list=sample_new
            
        

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        

        # Get the name and absolute path of the image to be read
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        frame_paths = os.path.join(self.videos_dir, video_id, f'{video_id}'+'_'+f'{frame_indices[0][-8:]}.png') #@whq
        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]
        

        
            # read the instance masks:
        annotation_paths = os.path.join(self.mask_annotations_dir, video_id, f'{frame_indices[0]}.png') 
        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)),dtype=torch.float32)
        all_object_indices = set().union(*[m.unique().tolist() for m in mask_annotations])
        all_object_indices.remove(0)  # remove the background index
        all_object_indices = sorted(list(all_object_indices))
        mask_annotations_by_object = []
        for obj_id in all_object_indices:
            obj_id_mask_annotations = torch.stack([(m == obj_id).to(torch.uint8) for m in mask_annotations])
            mask_annotations_by_object.append(obj_id_mask_annotations)
        mask_annotations_by_object = torch.stack(mask_annotations_by_object)
        #mask_annotations_by_frame = rearrange(mask_annotations_by_object, 'o t h w -> t o h w')
        mask_annotations_by_frame = mask_annotations_by_object  # o for object

            # next we get the referred instance index in the list of all the object ids:
        ref_obj_idx = torch.tensor(all_object_indices.index(int(text_query_dict['obj_id'])), dtype=torch.long)
        ref_mask_annotations=mask_annotations_by_frame[ref_obj_idx].unsqueeze(0)
        ref_mask_annotations=ref_mask_annotations.unsqueeze(0)

            # create a target dict for each frame:
        
        img=self.size_transforms(source_frames)
        mask_tensor = torch.nn.functional.interpolate(ref_mask_annotations, size=(224, 224), mode="nearest")
        mask=mask_tensor.squeeze(0)

        

        # Data enhancement process
        if self.subset_type=='train':
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
            
            if random.random() < 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)
                text_query = text_query.replace('bottom', '@').replace('bottom', 'top').replace('@', 'top')
                
        
        #index=self.text.index(text_query)
        ##text_feature=self.text_feature[index]
        #text_feature=torch.tensor(text_feature)
        #text_feature=text_feature.float()
        return img, mask, text_query,video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'


class ISICDataset(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,data_choice=0,transform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
        #self.videos_dir = path.join(dataset_path,subset_type, 'JPEGImages')
        self.videos_dir = '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/JPEGImages'
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        
        #self.mask_annotations_dir = path.join(dataset_path,subset_type, 'Annotations') 
        self.mask_annotations_dir= '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/Annotations'
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all/text.json'

        
        
        with open(metadata_file_path, 'r') as f:
            samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
        

        
        merge_set=[['seq_1','seq_9'],['seq_2','seq_10'],['seq_4','seq_12'],['seq_6','seq_14'],['seq_7','seq_15'],['seq_8','seq_16'],['seq_5','seq_13'],['seq_3','seq_11']]
        self.valid_dir=merge_set[data_choice]+merge_set[7-data_choice]
        print(merge_set[data_choice])
        print(merge_set[7-data_choice])
        train_dir=[['seq_1','seq_9'],['seq_2','seq_10'],['seq_4','seq_12'],['seq_6','seq_14'],['seq_7','seq_15'],['seq_8','seq_16'],['seq_5','seq_13'],['seq_3','seq_11']]
        train_dir.remove(merge_set[data_choice])
        train_dir.remove(merge_set[7-data_choice])
        
        
        self.train_dir=[]
        for i in train_dir:
            self.train_dir+=i 

        if self.subset_type=='train':
            data_dir=self.train_dir
        else:
            data_dir=self.valid_dir
        sample_new=[]
        print("train_dir:",self.train_dir)
        print("valid_dir",self.valid_dir)
        for i in range(len(samples_list)):
            if samples_list[i][0] in data_dir:
                sample_new.append(samples_list[i])
              
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = []
        transforms.append(T.Resize(size=(img_size, img_size)))
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)
        with open('/media/cygzz/data/guanjl/Code_self/text.pkl', 'rb') as f:
            all_info_80 = pickle.load(f)  
        self.f=all_info_80
        text=[]
        text_feature=[]
        for i in range(len(self.f)):
            item=self.f[i]
            text.append(item[0])
            text_feature.append(item[1])
        self.text=text
        self.text_feature=text_feature 
        pp=[]
        #self.samples_list=samples_list
        
        self.samples_list=sample_new
            
        

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):
        

        # Get the name and absolute path of the image to be read
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        frame_paths = os.path.join(self.videos_dir, video_id, f'{video_id}'+'_'+f'{frame_indices[0][-8:]}.png') #@whq
        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]
        

        
            # read the instance masks:
        annotation_paths = os.path.join(self.mask_annotations_dir, video_id, f'{frame_indices[0]}.png') 
        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)),dtype=torch.float32)
        all_object_indices = set().union(*[m.unique().tolist() for m in mask_annotations])
        all_object_indices.remove(0)  # remove the background index
        all_object_indices = sorted(list(all_object_indices))
        mask_annotations_by_object = []
        for obj_id in all_object_indices:
            obj_id_mask_annotations = torch.stack([(m == obj_id).to(torch.uint8) for m in mask_annotations])
            mask_annotations_by_object.append(obj_id_mask_annotations)
        mask_annotations_by_object = torch.stack(mask_annotations_by_object)
        #mask_annotations_by_frame = rearrange(mask_annotations_by_object, 'o t h w -> t o h w')
        mask_annotations_by_frame = mask_annotations_by_object  # o for object

            # next we get the referred instance index in the list of all the object ids:
        ref_obj_idx = torch.tensor(all_object_indices.index(int(text_query_dict['obj_id'])), dtype=torch.long)
        ref_mask_annotations=mask_annotations_by_frame[ref_obj_idx].unsqueeze(0)
        ref_mask_annotations=ref_mask_annotations.unsqueeze(0)

            # create a target dict for each frame:
        
        img=self.size_transforms(source_frames)
        mask_tensor = torch.nn.functional.interpolate(ref_mask_annotations, size=(224, 224), mode="nearest")
        mask=mask_tensor.squeeze(0)

        

        # Data enhancement process
        if self.subset_type=='train':
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
            
            if random.random() < 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)
                text_query = text_query.replace('bottom', '@').replace('bottom', 'top').replace('@', 'top')
                
        
        #index=self.text.index(text_query)
        ##text_feature=self.text_feature[index]
        #text_feature=torch.tensor(text_feature)
        #text_feature=text_feature.float()
        return img, mask, text_query,video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'
    


if __name__ == "__main__":


    # # with g_pathmgr.open("/media/cygzz/data/guanjl/Ref-EndoVis/Datasets/Ref-Endovis17/train/Meta/seq_1.json", "r") as f:
    # #             meta_annotation = json.load(f)
    # # print("finish")
    # data_choice=2
    # a_dir=merge_set.remove(merge_set[2])
    # train_dir.remove(merge_set[data_choice])
    # train_dir.remove(merge_set[7-data_choice])
    x=ISICDataset('train',dataset_path='/media/cygzz/data/guanjl/Code_self/datasets/Endovis-RS17all',data_choice=3)
    #x.__getitem__

    b=x[1]
    print('finish')
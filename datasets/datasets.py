import json
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torchvision.transforms.functional as F
from os import path
from glob import glob
from tqdm import tqdm

#import multiprocessing
from PIL import Image
import numpy as np
from einops import rearrange
import torchvision.transforms as T
import os
#from misc import nested_tensor_from_tensor_list,nested_tensor_from_videos_list
import pickle
class ReferRSVIS(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the full
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """                                                         #  data path
    def __init__(self, subset_type: str = 'train', dataset_path: str = '/home/whq/HKUSTGZ/RVOS/Dataset2/EndoVis-RS17', window_size=1):
        super(ReferRSVIS, self).__init__()
        assert subset_type in ['train', 'test'], "error, unsupported dataset subset type. use 'train' or 'test'."
        if subset_type == 'test':
            subset_type = 'valid'  # Refer-Youtube-VOS is tested on its 'validation' subset (see description above)
        self.subset_type = subset_type
        self.window_size = window_size
        num_videos_by_subset = {'train': 10, 'valid': 4}
        self.videos_dir = path.join(dataset_path, subset_type, 'JPEGImages')
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        if subset_type == 'train':
            self.mask_annotations_dir = path.join(dataset_path, subset_type, 'Annotations')  # only available for train
            # print(' len(glob(path.join(self.mask_annotations_dir, '*')))=', len(glob(path.join(self.mask_annotations_dir, '*'))))
            # print('num_videos_by_subset[subset_type]=',num_videos_by_subset[subset_type])
            # assert len(glob(path.join(self.mask_annotations_dir, '*'))) == num_videos_by_subset[subset_type], \
            #     f'error: {subset_type} subset is missing one or more mask annotations'
        else:
            self.mask_annotations_dir = None
        self.samples_list = self.generate_samples_metadata(dataset_path, subset_type)
        self.transforms = A2dSentencesTransforms(subset_type)
       # self.collator = Collator(subset_type)

    def generate_samples_metadata(self, dataset_path, subset_type):
        if subset_type == 'train':
            metadata_file_path = '/media/cygzz/data/guanjl/Code_self/train.json'
        else:  # validation
            metadata_file_path = '/media/cygzz/data/guanjl/Code_self/valid.json'
        if path.exists(metadata_file_path):
            print(f'loading {subset_type} subset samples metadata...')
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
                # print(samples_list)
                return samples_list
        
                
    def __getitem__(self, idx):
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        frame_paths = os.path.join(self.videos_dir, video_id, f'{frame_indices[0]}.png') #@whq
        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]

        

        if self.subset_type == 'train':
            # read the instance masks:
            annotation_paths = os.path.join(self.mask_annotations_dir, video_id, f'{frame_indices[0]}.png') 
            mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)))
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

            # create a target dict for each frame:
            targets = []
        
            for frame_masks in mask_annotations_by_frame:
                target = frame_masks
                target_in={'masks': frame_masks,
                          # idx in 'masks' of the text referred instance
                          'referred_instance_idx': ref_obj_idx,
                          # whether the referred instance is visible in the frame:
                          'is_ref_inst_visible': frame_masks[ref_obj_idx].any(),
                          'orig_size': frame_masks.shape[-2:],  # original frame shape without any augmentations
                          # size with augmentations, will be changed inside transforms if necessary
                          'size': frame_masks.shape[-2:],
                          'iscrowd': torch.zeros(len(frame_masks)),  # for compatibility with DETR COCO transforms
                          }
                targets.append(target_in)
        else:
            # validation subset has no annotations, so create dummy targets:
            targets = len(source_frames) * [None]

        source_frames, targets, text_query = self.transforms(source_frames, targets, text_query)
        target=targets[0]['masks']

        return source_frames, targets, text_query
        # else:  # validation:
        #     video_metadata = {'video_id': video_id,
        #                       'frame_indices': frame_indices,
        #                       'resized_frame_size': source_frames.shape[-2:],
        #                       'original_frame_size': original_frame_size,
        #                       'exp_id': text_query_dict['exp_id'],
        #                       'obj_id':text_query_dict['obj_id']}
        #     return source_frames, video_metadata, text_qu

    def __len__(self):
        return len(self.samples_list)


class A2dSentencesTransforms:
    def __init__(self, subset_type, horizontal_flip_augmentations=True, resize_and_crop_augmentations=True,
                 train_short_size=360, train_max_size=640, eval_short_size=360, eval_max_size=640, **kwargs):
        self.h_flip_augmentation = subset_type == 'train' and horizontal_flip_augmentations
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        scales = [train_short_size]  # size is slightly smaller than eval size below to fit in GPU memory
        transforms = []
        if resize_and_crop_augmentations:
            if subset_type == 'train':
                print('---------------------------------------------------------qqqqqqqqqqqqqqqqqqqqqqqqqqqqq')
                transforms.append(T.RandomResize(scales, max_size=train_max_size))
            elif subset_type == 'valid':
                transforms.append(T.RandomResize([eval_short_size], max_size=eval_max_size)),
        transforms.extend([T.ToTensor(), normalize])
        self.size_transforms = T.Compose(transforms)

    def __call__(self, source_frames, targets, text_query):
        if self.h_flip_augmentation and torch.rand(1) > 0.5:
            source_frames = [F.hflip(f) for f in source_frames]
            for t in targets:
                t['masks'] = F.hflip(t['masks'])
            # Note - is it possible for both 'right' and 'left' to appear together in the same query. hence this fix:
            text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
        source_frames, targets = list(zip(*[self.size_transforms(f, t) for f, t in zip(source_frames, targets)]))
        source_frames = torch.stack(source_frames)  # [T, 3, H, W]
        return source_frames, targets, text_query


# class Collator:
#     def __init__(self, subset_type):
#         self.subset_type = subset_type

#     def __call__(self, batch):
#         #f self.subset_type == 'train':
#             samples, targets, text_queries= list(zip(*batch))
#             samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
#             # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
#             targets = list(zip(*targets))
#             batch_dict = {
#                 'samples': samples,
#                 'targets': targets,
#                 'text_queries': text_queries,
                
#             }
#             return batch_dict
        # else:  # validation:
        #     samples, videos_metadata, text_queries, obj_w = list(zip(*batch))
        #     samples = nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
        #     batch_dict = {
        #         'samples': samples,
        #         'videos_metadata': videos_metadata,
        #         'text_queries': text_queries,
                
        #     }
        #     return batch_dict

import os
import torch
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

# 环境设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ISICDataset1(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,transform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
        #self.videos_dir = path.join(dataset_path,subset_type, 'JPEGImages')
        self.videos_dir1 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/train/JPEGImages'
        self.videos_dir2 = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/valid/JPEGImages'
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        
        #self.mask_annotations_dir = path.join(dataset_path,subset_type, 'Annotations') 
        self.mask_annotations_dir1= '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/train/Annotations'
        self.mask_annotations_dir2= '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/valid/Annotations'
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/'+subset_type+'.json'
        if subset_type=='valid':
            with open(metadata_file_path, 'r') as f:
                samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
            self.samples_list=samples_list
        
        if subset_type=='train':
            with open('/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/train.json', 'r') as f:
                samples_list1 = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
            
            with open('/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/valid.json', 'r') as f:
                samples_list2 = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
            
            self.samples_list=samples_list1+samples_list2
        
        
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
        
        #self.samples_list=samples_list
            
        

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        # Get the name and absolute path of the image to be read
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        if os.path.exists(os.path.join(self.videos_dir1, video_id, f'{video_id}'+'_'+f'{frame_indices[0][-8:]}.png') ):
            frame_paths = os.path.join(self.videos_dir1, video_id, f'{video_id}'+'_'+f'{frame_indices[0][-8:]}.png') #@whq
        else:
            frame_paths = os.path.join(self.videos_dir2, video_id, f'{video_id}'+'_'+f'{frame_indices[0][-8:]}.png')

        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]
        

        
            # read the instance masks:
        if os.path.exists(os.path.join(self.mask_annotations_dir1, video_id, f'{frame_indices[0]}.png') ):
            annotation_paths = os.path.join(self.mask_annotations_dir1, video_id, f'{frame_indices[0]}.png') 
        else:
            annotation_paths = os.path.join(self.mask_annotations_dir2, video_id, f'{frame_indices[0]}.png') 

        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)))
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
        if self.subset_type=='train_mini' or self.subset_type=='train':
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
                
        
        #index=self.text.index(text_query)
        ##text_feature=self.text_feature[index]
        #text_feature=torch.tensor(text_feature)
        #text_feature=text_feature.float()
        return img, mask, text_query,video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'
    

class ISICDataset(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,transform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
    
        self.videos_dir = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/'+subset_type+'/JPEGImages'
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'

        self.mask_annotations_dir= '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/'+subset_type+'/Annotations'
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18/'+subset_type+'.json'
        
        with open(metadata_file_path, 'r') as f:
            samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
        
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
        
        self.samples_list=samples_list
            
        

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
        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)))
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
        if self.subset_type=='train_mini' or self.subset_type=='train':
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
                
        
        #index=self.text.index(text_query)
        ##text_feature=self.text_feature[index]
        #text_feature=torch.tensor(text_feature)
        #text_feature=text_feature.float()
        return img, mask, text_query,video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'

class ISICDataset_sota(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,transform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
        self.videos_dir = path.join(dataset_path,subset_type, 'JPEGImages')
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        
        self.mask_annotations_dir = path.join(dataset_path,subset_type, 'Annotations') 
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/'+subset_type+'.json'
        
        with open(metadata_file_path, 'r') as f:
            samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
        
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms = []
        transforms.append(T.Resize(size=(360, 640)))
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
        
        self.samples_list=samples_list
            
        

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        # Get the name and absolute path of the image to be read
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        frame_paths = os.path.join(self.videos_dir, video_id, f'{frame_indices[0]}.png') #@whq
        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]
        

        
            # read the instance masks:
        annotation_paths = os.path.join(self.mask_annotations_dir, video_id, f'{frame_indices[0]}.png') 
        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)))
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
        mask_tensor = torch.nn.functional.interpolate(ref_mask_annotations, size=(360, 640), mode="nearest")
        mask=mask_tensor.squeeze(0)

        

        # Data enhancement process
        if self.subset_type=='train_mini' or self.subset_type=='train':
            # Save the randomized state so that the same transforms will be applied to masks
            # and images when finer transforms are used
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                text_query = text_query.replace('left', '@').replace('right', 'left').replace('@', 'right')
                
        
        index=self.text.index(text_query)
        text_feature=self.text_feature[index]
        text_feature=torch.tensor(text_feature)
        text_feature=text_feature.float()
        return img, mask, text_query,video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'


class ISICDataset_eval(Dataset):
    def __init__(self, subset_type='train',dataset_path='',img_size=224,ransform=None):

        # Read txt file
        self.subset_type = subset_type
        self.img_size=img_size
        
        num_videos_by_subset = {'train': 10, 'valid': 4}
        self.videos_dir = path.join(dataset_path, subset_type, 'JPEGImages')
        # assert len(glob(path.join(self.videos_dir, '*'))) == num_videos_by_subset[subset_type], \
        #     f'error: {subset_type} subset is missing one or more frame samples'
        
        self.mask_annotations_dir = path.join(dataset_path,subset_type , 'Annotations') 
        
        metadata_file_path = '/media/cygzz/data/guanjl/Code_self/'+subset_type+'.json'
        
        with open(metadata_file_path, 'r') as f:
            samples_list = [tuple(a) for a in tqdm(json.load(f), disable=False and dist.get_rank() != 0)]
        
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
        
        
        self.samples_list=pp
            
        

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, idx):

        # Get the name and absolute path of the image to be read
        video_id, frame_indices, text_query_dict = self.samples_list[idx]
        text_query = text_query_dict['exp']
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        # frame_paths = [path.join(self.videos_dir, video_id, f'{idx}.jpg') for idx in frame_indices]
        frame_paths = os.path.join(self.videos_dir, video_id, f'{frame_indices[0]}.png') #@whq
        source_frames = Image.open(frame_paths).convert('RGB')
        original_frame_size = source_frames.size[::-1]
        

        
            # read the instance masks:
        annotation_paths = os.path.join(self.mask_annotations_dir, video_id, f'{frame_indices[0]}.png') 
        mask_annotations = torch.tensor(np.array(Image.open(annotation_paths)))
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
        
            
        index=self.text.index(text_query)
        text_feature=self.text_feature[index]
        text_feature=torch.tensor(text_feature)
        text_feature=text_feature.float()
        return img, mask, text_feature, video_id+"_obj"+text_query_dict['obj_id']+"_"+f'{frame_indices[0]}.png'
    
class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        """ Generic dataset for loading npy files, npy stores 3D arrays, channel 0 is the image, channel 1 is the label.
        """

        super().__init__()
        self.transform = transform
        self.test_flag = test_flag
        self.directory = os.path.expanduser(directory)
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith('.npy')]

    def __getitem__(self, x: int):
        # Access to the name of the document
        file_name = self.filenames[x]

        # Read file
        npy_img = np.load(os.path.join(self.directory, file_name))

        # 获取图像, 并进行维度交换 通道在前
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Get the mask and binarize it
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)

        # Format conversion of images and masks
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        if self.transform:
            # Save the randomized state so that the same transforms will be applied to the mask and image
            # when finer transforms are used
            state = torch.get_rng_state()
            torch.set_rng_state(state)

            # Data augmentation changes
            image = self.transform(image)
            mask = self.transform(mask)

        # Return image and mask
        if self.test_flag:
            return image, mask, file_name
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)


if __name__ == "__main__":


    # with g_pathmgr.open("/media/cygzz/data/guanjl/Ref-EndoVis/Datasets/Ref-Endovis17/train/Meta/seq_1.json", "r") as f:
    #             meta_annotation = json.load(f)
    # print("finish")
    train_dataset=ISICDataset('train','/media/cygzz/data/guanjl/Code_self/datasets/EndoVis-RS18')
    for j in train_dataset:
        print(j[2]," ",j[3])
    print('finish')
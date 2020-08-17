# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:50:11 2020

@author: Swaminaathan P J M
"""

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import imageio
import fnmatch
import tqdm
import glob

class DEMO(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        #self.real = glob.glob(path + '/real/*')
        self.real = glob.glob(path + 'real/*')
        self.fake = glob.glob(path + 'fake/*')
        self.real_videos=[]
        self.real_videos_OF=[]
        
        for i in self.real:
            j=(i.split('/')[8])
            if not fnmatch.fnmatch(j, '*OF'):
                self.videos=glob.glob(path+'real/'+j+'/*')
                for f in self.videos:
                    temp=[]
                    temp.append(f)
                    temp.append(np.array([0]))
                    self.real_videos.append(temp)
            elif fnmatch.fnmatch(j, '*OF'):
                self.videos=glob.glob(path+'real/'+j+'/*')
                for f in self.videos:
                    temp=[]
                    temp.append(f)
                    temp.append(np.array([0]))
                    self.real_videos_OF.append(temp)
        print(len(self.real_videos))
        print(len(self.real_videos_OF))

        for i in self.fake:
            j=(i.split('/')[8])
            if not fnmatch.fnmatch(j, '*OF'):
                self.videos=glob.glob(path+'fake/'+j+'/*')
                for f in self.videos:
                    temp=[]
                    temp.append(f)
                    temp.append(np.array([1]))
                    self.real_videos.append(temp)
            elif fnmatch.fnmatch(j, '*OF'):
                self.videos=glob.glob(path+'fake/'+j+'/*')
                for f in self.videos:
                    temp=[]
                    temp.append(f)
                    temp.append(np.array([1]))
                    self.real_videos_OF.append(temp)

        print(len(self.real_videos))
        print(len(self.real_videos_OF))

        self.real_videos = sorted(self.real_videos)
        self.real_videos_OF = sorted(self.real_videos_OF)
        self.num_data1=len(self.real_videos)

    def __getitem__(self, index):
        image1 = Image.open(self.real_videos[index][0])
        image2 = Image.open(self.real_videos_OF[index][0])
        #return (self.transform(image1), self.transform(image2), torch.FloatTensor(self.real_videos[index][1]))
        return (self.transform(image1), self.transform(image2), torch.FloatTensor(self.real_videos[index][1]),self.real_videos[index][0],self.real_videos_OF[index][0])

    def __len__(self):
        return self.num_data1


def get_loader(crop_size,
               image_size,
               batch_size,
               gpus,
               nodes,
               nrank,
               rank,
               num_workers=0,
               verbose=False,
               demo=True):
    """Build and return data loader."""
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[1.0, 1.0, 1.0])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=Image.ANTIALIAS),
        transforms.ToTensor(),
        normalize,
    ])
    batch_size = batch_size
    dist_world_size = gpus * nodes  
    dataset = DEMO(demo, transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=dist_world_size,rank=rank)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,     
        pin_memory=True,
        sampler=train_sampler)
    return data_loader

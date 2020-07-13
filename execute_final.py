# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:59:34 2020

@author: Swaminaathan P J M
"""

import wandb
import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import os
import GPUtil
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
import numpy as np
import math
from data_loader_final import get_loader
from torch.autograd import Variable
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

#VGG Network
def make_layers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# VGG-16
cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

def vgg16(pretrained='', **kwargs):
    model = VGG_IMAGE(make_layers(cfg['D']), **kwargs)
    return model

#Class Definition for VGG
class VGG_IMAGE(nn.Module):
    def __init__(self, features, num_classes=2):
        super(VGG_IMAGE, self).__init__()
        self.features = features
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

#Binary Classification
class classification(nn.Module):

    #Global Attention Pooling and FC layers
    def __init__(self):
        super(classification, self).__init__()
        self.aggregation=nn.Sequential(
            nn.Linear(512*14*7, 1),
            nn.Softmax(dim=1))
        self.binclassifier= nn.Sequential(
            nn.Linear(512*14*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1000,512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1))

    def forward(self, xtotal):
        x1=self.aggregation(xtotal)
        x2=x1*xtotal
        x3=torch.sum(x2, dim=1)
        x4=self.binclassifier(x3)
        return x4
    
#Execution Part
class Solve(object):
    def __init__(self, config):
       
        # Model Initialization
        self.model1=None
        self.AU01_model=None
        self.AU02_model=None
        self.AU04_model=None
        self.AU06_model=None
        self.AU07_model=None
        self.AU10_model=None
        self.AU12_model=None
        self.AU14_model=None
        self.AU15_model=None
        self.AU17_model=None
        self.AU23_model=None
        self.AU24_model=None
    
    #Function for converting to Cuda
    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    #Training Part  
    def training(self,gpu,config):

        #Configuring Distributed Data Parallelism
        dist_world_size = config.gpus * config.nodes  
        os.environ['MASTER_ADDR'] = 'localhost'      
        os.environ['MASTER_PORT'] = '12853'
        rank = config.nr * config.gpus + gpu	    
        dist.init_process_group(backend='nccl',init_method='env://', world_size=dist_world_size, rank=rank)  
        torch.manual_seed(0)
        torch.cuda.set_device(gpu)

        #Initialsing AU models
        self.AU01_model=vgg16(num_classes=1)
        self.AU02_model=vgg16(num_classes=1)
        self.AU04_model=vgg16(num_classes=1)
        self.AU06_model=vgg16(num_classes=1)
        self.AU07_model=vgg16(num_classes=1)
        self.AU10_model=vgg16(num_classes=1)
        self.AU12_model=vgg16(num_classes=1)
        self.AU14_model=vgg16(num_classes=1)
        self.AU15_model=vgg16(num_classes=1)
        self.AU17_model=vgg16(num_classes=1)
        self.AU23_model=vgg16(num_classes=1)
        self.AU24_model=vgg16(num_classes=1)
        
        #Loading AU weights
        self.AU01_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU01.pth'))
        self.AU02_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU02.pth'))
        self.AU04_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU04.pth'))
        self.AU06_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU06.pth'))
        self.AU07_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU07.pth'))
        self.AU10_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU10.pth'))
        self.AU12_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU12.pth'))
        self.AU14_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU14.pth'))
        self.AU15_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU15.pth'))
        self.AU17_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU17.pth'))
        self.AU23_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU23.pth'))
        self.AU24_model.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/AU24.pth'))

        #AU - GPU
        self.AU01_model.cuda(gpu)
        self.AU02_model.cuda(gpu)
        self.AU04_model.cuda(gpu)
        self.AU06_model.cuda(gpu)
        self.AU07_model.cuda(gpu)
        self.AU10_model.cuda(gpu)
        self.AU12_model.cuda(gpu)
        self.AU14_model.cuda(gpu)
        self.AU15_model.cuda(gpu)
        self.AU17_model.cuda(gpu)
        self.AU23_model.cuda(gpu)
        self.AU24_model.cuda(gpu)

        #AU - Distributed Data Parallelism
        self.AU01_model = nn.parallel.DistributedDataParallel(self.AU01_model,device_ids=[gpu])
        self.AU02_model = nn.parallel.DistributedDataParallel(self.AU02_model,device_ids=[gpu])
        self.AU04_model = nn.parallel.DistributedDataParallel(self.AU04_model,device_ids=[gpu])
        self.AU06_model = nn.parallel.DistributedDataParallel(self.AU06_model,device_ids=[gpu])
        self.AU07_model = nn.parallel.DistributedDataParallel(self.AU07_model,device_ids=[gpu])
        self.AU10_model = nn.parallel.DistributedDataParallel(self.AU10_model,device_ids=[gpu])
        self.AU12_model = nn.parallel.DistributedDataParallel(self.AU12_model,device_ids=[gpu])
        self.AU14_model = nn.parallel.DistributedDataParallel(self.AU14_model,device_ids=[gpu])
        self.AU15_model = nn.parallel.DistributedDataParallel(self.AU15_model,device_ids=[gpu])
        self.AU17_model = nn.parallel.DistributedDataParallel(self.AU17_model,device_ids=[gpu])
        self.AU23_model = nn.parallel.DistributedDataParallel(self.AU23_model,device_ids=[gpu])
        self.AU24_model = nn.parallel.DistributedDataParallel(self.AU24_model,device_ids=[gpu])

        #Freeze weights of AU Model
        for param in self.AU01_model.parameters():
            param.requires_grad = False
        for param in self.AU02_model.parameters():
            param.requires_grad = False
        for param in self.AU04_model.parameters():
            param.requires_grad = False
        for param in self.AU06_model.parameters():
            param.requires_grad = False
        for param in self.AU07_model.parameters():
            param.requires_grad = False
        for param in self.AU10_model.parameters():
            param.requires_grad = False
        for param in self.AU12_model.parameters():
            param.requires_grad = False
        for param in self.AU14_model.parameters():
            param.requires_grad = False
        for param in self.AU15_model.parameters():
            param.requires_grad = False
        for param in self.AU17_model.parameters():
            param.requires_grad = False
        for param in self.AU23_model.parameters():
            param.requires_grad = False
        for param in self.AU24_model.parameters():
            param.requires_grad = False
        

        self.model1=classification()  
        self.model1.cuda(gpu)
        self.model1 = nn.parallel.DistributedDataParallel(self.model1,device_ids=[gpu])

        #Data Loader
        img_size = config.image_size
        self.rgb_loader = get_loader(img_size,
                                img_size,
                                config.batch_size,
                                config.gpus,
                                config.nodes,
                                config.nr,
                                rank,
                                demo=config.DEMO,
                                num_workers=config.num_workers,
                                verbose=True)

        #Loading trained model you have for classifiacation
        self.model1.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/AUNETS_final/model_Relu_distributed_full_3.pth'))

        wandb.init(project="deepfakes",reinit=True)
        wandb.watch(self.model1)

        pos_weight = torch.Tensor([(1 / 10)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        learning_rate=0.0001
        optimizer = optim.SGD(self.model1.parameters(),lr=learning_rate)

        for epoch in range(2): 
            if(epoch!=0 and epoch%2==0):
                learning_rate*=0.1
            optimizer.zero_grad()
            epoch_loss=0.0
            i=0
            
            for real_rgb in self.rgb_loader:
                real_im = self.to_var(real_rgb[0], volatile=True)
                real_of = self.to_var(real_rgb[1], volatile=True)
                img_of = torch.cat([real_im, real_of], dim=3)
                
                xt=[]

                x01 = self.AU01_model.module.features(img_of)
                x01 = x01.view(x01.size(0),-1)
                xt.append(x01)

                x02 = self.AU02_model.module.features(img_of)
                x02 = x02.view(x02.size(0),-1)
                xt.append(x02)

                x04 = self.AU04_model.module.features(img_of)
                x04 = x04.view(x04.size(0),-1)
                xt.append(x04)

                x06 = self.AU06_model.module.features(img_of)
                x06 = x06.view(x06.size(0),-1)
                xt.append(x06)

                x07 = self.AU07_model.module.features(img_of)
                x07 = x07.view(x07.size(0),-1)
                xt.append(x07)

                x10 = self.AU10_model.module.features(img_of)
                x10 = x10.view(x10.size(0),-1)
                xt.append(x10)

                x12 = self.AU12_model.module.features(img_of)
                x12 = x12.view(x12.size(0),-1)
                xt.append(x12)

                x14 = self.AU14_model.module.features(img_of)
                x14 = x14.view(x14.size(0),-1)
                xt.append(x14)

                x15 = self.AU15_model.module.features(img_of)
                x15 = x15.view(x15.size(0),-1)
                xt.append(x15)

                x17 = self.AU17_model.module.features(img_of)
                x17 = x17.view(x17.size(0),-1)
                xt.append(x17)

                x23 = self.AU23_model.module.features(img_of)
                x23 = x23.view(x23.size(0),-1)
                xt.append(x23)

                x24 = self.AU24_model.module.features(img_of)
                x24 = x24.view(x24.size(0),-1)
                xt.append(x24)

                GPUtil.showUtilization()
                result = torch.stack(xt, dim=1)
                out_temp = self.model1(result)
                out_temp2=out_temp.cpu()

                #Loss
                loss=criterion(out_temp2, torch.FloatTensor(real_rgb[2])) 
                loss.backward()
                optimizer.step()
                
                epoch_loss += out_temp.shape[0] * loss.item()
                i+=1
                
                print(loss)
                if(rank%config.gpus==0 and i%1000==0):
                    torch.save(self.model1.state_dict(),'/projects/academic/doermann/Deepfakes/AUNETS_final/model_Relu_distributed_full_4.pth')
                wandb.log({"Loss": loss})
            print("Epoch: {},Epoch Loss: {}" .format(epoch,epoch_loss/(i*64)))
            
            #Saving Model
            if(rank%config.gpus==0):
                torch.save(self.model1.state_dict(),'/projects/academic/doermann/Deepfakes/AUNETS_final/model_Relu_distributed_full_4.pth')
            wandb.log({"Epoch": epoch, "Epoch_Loss": (epoch_loss/(i*64))})

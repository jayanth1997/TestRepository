import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import GlobalAttentionPooling
from functools import partial
from model import BaseRGCN
import wandb
import torch
import cv2
import glob
from collections import defaultdict
import fnmatch
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
from data_loader_graph import get_loader
from torch.autograd import Variable
import warnings
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

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

#0 2 5 7 10 12 14 17 19 21 24 26 28
cfg = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

def vgg16(pretrained='', **kwargs):
    model = VGG_IMAGE(make_layers(cfg['D']), **kwargs)
    return model

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


class EntityClassify(BaseRGCN):
    def build_input_layer(self):
        return RelGraphConv(self.num_nodes, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)

    def build_output_layer(self):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=None, self_loop=self.use_self_loop)

class Solver(object):
    def __init__(self, config):
        # Model
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
    
    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)
        
    def training(self,gpu,config):
        dist_world_size = config.gpus * config.nodes  
        os.environ['MASTER_ADDR'] = 'localhost'      
        os.environ['MASTER_PORT'] = '12853'
        rank = config.nr * config.gpus + gpu	    
        print("jump")                      
        dist.init_process_group(backend='nccl',init_method='env://', world_size=dist_world_size, rank=rank)  
        torch.manual_seed(0)
        print("dub")
        torch.cuda.set_device(gpu)
        print("hello3")
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        
        img_size = config.image_size
        self.val_loader = get_loader(img_size,
                                img_size,
                                config.batch_size,
                                config.gpus,
                                config.nodes,
                                config.nr,
                                rank,
                                demo=config.DEMO,
                                num_workers=config.num_workers,
                                verbose=True)

        wandb.init(project="graph",reinit=True)
        #wandb.watch(self.model1)

        num_nodes=12
        num_rels=144
        num_bases=-1
        num_classes=2
        num_hidden_layers=1
        dropout=0
        use_cuda = torch.cuda.is_available()

        print("ASASASASASAS")
        self.model1 = EntityClassify(1568,
                                     512,
                                     num_classes,
                                     num_rels,
                                     num_bases=num_bases,
                                     num_hidden_layers=num_hidden_layers,
                                     dropout=dropout,
                                     use_self_loop=True,
                                     use_cuda=use_cuda)
        print("zxzxzxzxzxzxzxzx")
        print(self.model1)
        self.model1=self.model1.to(gpu)
        self.model1 = nn.parallel.DistributedDataParallel(self.model1,device_ids=[gpu],find_unused_parameters=True)
        self.model1.load_state_dict(torch.load('/projects/academic/doermann/Deepfakes/RGCN/graph_model_5.pth'))
        self.model1=self.model1.to(gpu)
        wandb.watch(self.model1)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Number of trainable params ', count_parameters(self.model1))

        

        tot_ac=[]
        dic={}
        #dic = defaultdict(list)
        path=config.DEMO
        self.real = glob.glob(path + 'real/*')
        self.fake = glob.glob(path + 'fake/*')
        
        for i in self.real:
            j=(i.split('/')[8])
            if not fnmatch.fnmatch(j, '*OF'):
                dic[j]=[0,0]
        for i in self.fake:
            j=(i.split('/')[8])
            if not fnmatch.fnmatch(j, '*OF'):
                dic[j]=[0,0]

        print(dic)
        tp=0
        fn=0
        fp=0
        tn=0
        with torch.no_grad():
            print("PJM")
            for val_rgb in self.val_loader:
                val_im = self.to_var(val_rgb[0], volatile=True)
                val_of = self.to_var(val_rgb[1], volatile=True)
                img_val_of = torch.cat([val_im, val_of], dim=3)

                xv=[]
                #print(img_of.shape)
                x01v = self.AU01_model.module.features(img_val_of)
                x01v = x01v.view(x01v.size(0),-1)
                xv.append(x01v)

                x02v = self.AU02_model.module.features(img_val_of)
                x02v = x02v.view(x02v.size(0),-1)
                xv.append(x02v)

                x04v = self.AU04_model.module.features(img_val_of)
                x04v = x04v.view(x04v.size(0),-1)
                xv.append(x04v)

                x06v = self.AU06_model.module.features(img_val_of)
                x06v = x06v.view(x06v.size(0),-1)
                xv.append(x06v)

                x07v = self.AU07_model.module.features(img_val_of)
                x07v = x07v.view(x07v.size(0),-1)
                xv.append(x07v)

                x10v = self.AU10_model.module.features(img_val_of)
                x10v = x10v.view(x10v.size(0),-1)
                xv.append(x10v)

                x12v = self.AU12_model.module.features(img_val_of)
                x12v = x12v.view(x12v.size(0),-1)
                xv.append(x12v)

                x14v = self.AU14_model.module.features(img_val_of)
                x14v = x14v.view(x14v.size(0),-1)
                xv.append(x14v)

                x15v = self.AU15_model.module.features(img_val_of)
                x15v = x15v.view(x15v.size(0),-1)
                xv.append(x15v)

                x17v = self.AU17_model.module.features(img_val_of)
                x17v = x17v.view(x17v.size(0),-1)
                xv.append(x17v)

                x23v = self.AU23_model.module.features(img_val_of)
                x23v = x23v.view(x23v.size(0),-1)
                xv.append(x23v)

                x24v = self.AU24_model.module.features(img_val_of)
                x24v = x24v.view(x24v.size(0),-1)
                xv.append(x24v)
                res = torch.stack(xv, dim=1)

                adj_matrix=np.ones((12,12))
                degs=np.sum(adj_matrix, axis=1)
                degs[degs==0]=1
                adj_matrix=adj_matrix /degs[:, None]
                sp_matrix=sparse.coo_matrix(adj_matrix)
                norm=sp_matrix.data
                edge_norm=torch.FloatTensor(norm)

                a=np.arange(num_rels)
                edge_type = torch.from_numpy(a).long()
                edge_type=edge_type.view(-1)


                #For batch size = 1
                g=dgl.DGLGraph()
                g.add_nodes(12)
                for m in range(12):
                    for n in range(12):
                        g.add_edge(m, n)


                feats=res[0]
                g.ndata['node_features']=torch.tensor(feats)
                g.ndata['node_features']=g.ndata['node_features'].cuda()


                g.edata['type']=edge_type
                g.edata['type']=g.edata['type'].cuda()

                g.edata['norm']=edge_norm
                g.edata['norm']=g.edata['norm'].cuda()

                #For batch size > 1
                for k in range(1,val_rgb[0].shape[0]):   
                    g1=dgl.DGLGraph()
                    g1.add_nodes(12)
                    for m in range(12):
                        for n in range(12):
                            g1.add_edge(m, n)
                    feats=res[k]
                    g1.ndata['node_features']=torch.tensor(feats)
                    g1.ndata['node_features']=g1.ndata['node_features'].cuda()
                    g1.edata['type']=edge_type
                    g1.edata['type']=g1.edata['type'].cuda()
                    g1.edata['norm']=edge_norm
                    g1.edata['norm']=g1.edata['norm'].cuda()
                    g=dgl.batch([g,g1])  
                            
                logits_v = self.model1(g, g.ndata['node_features'], g.edata['type'], g.edata['norm'])
                out_temp2_v=logits_v.cpu()
                out_temp_val_sig=F.sigmoid(out_temp2_v)
                temp_v=torch.round(out_temp_val_sig)


                for i in range(val_rgb[0].shape[0]):
                    #if(out_temp[i]==1):
                        #dic[real_rgb[3][i].split('/')[8]]=1
                    
                    if(temp_v[i]==0):
                        dic[val_rgb[3][i].split('/')[8]][0]+=1
                    elif(temp_v[i]==1):
                        dic[val_rgb[3][i].split('/')[8]][1]+=1

                    if(temp_v[i]==0 and val_rgb[2][i]==0):
                        tp+=1
                    elif(temp_v[i]==1 and val_rgb[2][i]==0):
                        fn+=1
                    elif(temp_v[i]==0 and val_rgb[2][i]==1):
                        fp+=1
                    elif(temp_v[i]==1 and val_rgb[2][i]==1):
                        tn+=1


                correct_v = (temp_v == val_rgb[2]).float().sum()
                acc_v=correct_v/val_rgb[0].shape[0]
                print("Accuracy: {}" .format(acc_v))
                tot_ac.append(acc_v)
                wandb.log({"Testing Accuracy": acc_v})

        print(tot_ac)
        tot_val_ac=torch.mean(torch.stack(tot_ac), dim=0)
        wandb.log({"Total Testing Accuracy": tot_val_ac})




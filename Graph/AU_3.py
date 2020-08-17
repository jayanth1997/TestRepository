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
        
        self.AU01_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU01.pth'))
        self.AU02_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU02.pth'))
        self.AU04_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU04.pth'))
        self.AU06_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU06.pth'))
        self.AU07_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU07.pth'))
        self.AU10_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU10.pth'))
        self.AU12_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU12.pth'))
        self.AU14_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU14.pth'))
        self.AU15_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU15.pth'))
        self.AU17_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU17.pth'))
        self.AU23_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU23.pth'))
        self.AU24_model.load_state_dict(torch.load('/projects/academic/doermann/PJM/AUNETS_final/AU24.pth'))

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

        if(config.val==True):
            self.val_loader = get_loader(img_size,
                                    img_size,
                                    config.batch_size,
                                    config.gpus,
                                    config.nodes,
                                    config.nr,
                                    rank,
                                    demo=config.DEMO2,
                                    num_workers=config.num_workers,
                                    verbose=True)


        wandb.init(project="graph",reinit=True)

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
        self.model1.load_state_dict(torch.load('/projects/academic/doermann/PJM/RGCN/graph_model_6.pth'))
        wandb.watch(self.model1)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('Number of trainable params ', count_parameters(self.model1))

        
        pos_weight = torch.Tensor([(1/10)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #criterion = nn.BCEWithLogitsLoss()
        learning_rate=0.00001
        optimizer = optim.Adam(self.model1.parameters(), lr=learning_rate, weight_decay=5e-5)


        for epoch in range(1): 
            epoch_loss=0.0
            i=0
            print("hello5")
            accu=[]
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
                #print(result.shape)

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


                feats=result[0]
                g.ndata['node_features']=torch.tensor(feats)
                g.ndata['node_features']=g.ndata['node_features'].cuda()


                g.edata['type']=edge_type
                g.edata['type']=g.edata['type'].cuda()

                g.edata['norm']=edge_norm
                g.edata['norm']=g.edata['norm'].cuda()

                #For batch size > 1
                for k in range(1,real_rgb[0].shape[0]):   
                    g1=dgl.DGLGraph()
                    g1.add_nodes(12)
                    for m in range(12):
                        for n in range(12):
                            g1.add_edge(m, n)
                    feats=result[k]
                    g1.ndata['node_features']=torch.tensor(feats)
                    g1.ndata['node_features']=g1.ndata['node_features'].cuda()
                    g1.edata['type']=edge_type
                    g1.edata['type']=g1.edata['type'].cuda()
                    g1.edata['norm']=edge_norm
                    g1.edata['norm']=g1.edata['norm'].cuda()
                    g=dgl.batch([g,g1])  
 
                logits = self.model1(g, g.ndata['node_features'], g.edata['type'], g.edata['norm'])
                out_temp2=logits.cpu()
                loss=criterion(out_temp2, torch.FloatTensor(real_rgb[2]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                out_temp_sig=F.sigmoid(out_temp2)
                print(real_rgb[2])
                print(out_temp_sig)
                #print(loss.item())
                epoch_loss += logits.shape[0] * loss.item()
                i+=1
                temp = torch.round(out_temp_sig)
                correct = (temp == real_rgb[2]).float().sum()
                acc=correct/real_rgb[0].shape[0]
                accu.append(acc)
                if(i%10000==0):
                    tot_train_ac=torch.mean(torch.stack(accu), dim=0)
                    print("Accuracy: {}".format(tot_train_ac))
                    wandb.log({"Inter Accuracy":tot_train_ac})

                print("Training Loss: {}".format(loss))
                print("Training Accuracy: {}".format(acc))
                wandb.log({"Training Loss": loss})
                wandb.log({"Training Accuracy": acc})
                if(rank%config.gpus==0 and i%5000==0):
                    torch.save(self.model1.state_dict(),'/projects/academic/doermann/PJM/RGCN/graph_model_7.pth')

                tot_ac=[]
                loss_tot=[]
                if(config.val==True):
                    if(i%25000==0):
                        print(">>>>>>>>>>>>>>>>>>VALIDATION SET<<<<<<<<<<<<<<<<<<<<<")
                        with torch.no_grad():
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
                                loss_v=criterion(out_temp2_v, torch.FloatTensor(val_rgb[2])) 
                                out_temp_val_sig=F.sigmoid(out_temp2_v)
                                temp_v=torch.round(out_temp_val_sig)
                                correct_v = (temp_v == val_rgb[2]).float().sum()
                                acc_v=correct_v/val_rgb[0].shape[0]
                                tot_ac.append(acc_v)
                                loss_tot.append(loss_v)
                                print("Validation Loss: {}".format(loss_v))
                                print("Validation Accuracy: {}".format(acc_v))
                                wandb.log({"Validation Loss": loss_v})
                                wandb.log({"Validation Accuracy": acc_v})

                            print(loss_tot)
                            print(tot_ac)
                            loss_val=torch.mean(torch.stack(loss_tot), dim=0)
                            tot_val_ac=torch.mean(torch.stack(tot_ac), dim=0)
                            print("Total Validation Loss: {}".format(loss_val))
                            print("Total Validation Accuracy: {}".format(tot_val_ac))
                            wandb.log({"Total Validation Loss": loss_val})
                            wandb.log({"Total Validation Accuracy": tot_val_ac})

            tot_train_ac=torch.mean(torch.stack(accu), dim=0)
            print("Epoch: {},Training Epoch Loss: {}, Accuracy: {}" .format(epoch,epoch_loss/(i*4),tot_train_ac))
            wandb.log({"Epoch": epoch, "Epoch_Loss": (epoch_loss/(i*4)),"Accuracy":tot_train_ac})
            if(rank%config.gpus==0):
                torch.save(self.model1.state_dict(),'/projects/academic/doermann/PJM/RGCN/graph_model_7.pth')



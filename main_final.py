# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:38:59 2020

@author: Swaminaathan P J M
"""

#!/d/Anaconda/envs/gpu/Scripts/ipython
import argparse
import os
import sys
from data_loader_final import get_loader
import torch.multiprocessing as mp
from torch.backends import cudnn
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    # Data_path
    parser.add_argument('--data_path', type=str, default='')
    #parser.add_argument('--val', action='store_true', default=False)
    #parser.add_argument('--DEMO2', type=str, default='')

    # Distributed Training
    parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    config = parser.parse_args()
    print(config)
        
    from execute_final import Solve
    solve = Solve(config)
    mp.spawn(solve.training, nprocs=config.gpus, args=(config,))


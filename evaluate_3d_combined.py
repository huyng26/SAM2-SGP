# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
import wandb
from datetime import datetime
import pytz
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def evaluate(rank=0, world_size=0):

    args = cfg.parse_args()

    if args.distributed:
        setup(rank, world_size)
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    if args.wandb_enabled:
        wandb.init(project="Staged_MedSAM2", name=args.exp_name)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(dtype=torch.bfloat16)

    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights["model"])
    
    if args.distributed:
        net = DDP(net, device_ids=[rank])
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    nice_train_loader, nice_test_loader = get_dataloader(args)
    
    net.eval()

    loss, iou, dice = function.validation_sam(args, nice_test_loader, 0, net, rank=rank)

    if args.distributed:
        dist.all_reduce(loss), dist.all_reduce(iou), dist.all_reduce(dice)
        loss, iou, dice = loss/world_size, iou/world_size, dice/world_size
        print(f"val/loss: {loss}, val/IOU: {iou}, val/dice : {dice}")
    else:
        print(f"val/loss: {loss}, val/IOU: {iou}, val/dice : {dice}")
    
    if args.wandb_enabled:
        wandb.log({'val/loss': loss,'val/IOU' : iou, 'val/dice' : dice})
            
    if args.distributed:
        cleanup()         

def main():
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = cfg.parse_args()
    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        evaluate()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3

""" evaluate network using pytorch
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
    
    # Check if distributed is actually enabled (not "none" string)
    is_distributed = args.distributed not in [False, 'none', 'None', None]

    if is_distributed:
        setup(rank, world_size)
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    # Initialize wandb - only on rank 0 for distributed training
    wandb_enabled = getattr(args, 'wandb_enabled', False)
    if wandb_enabled and (not is_distributed or rank == 0):
        # Get timezone-aware timestamp
        try:
            tz = pytz.timezone('Asia/Ho_Chi_Minh')  # Adjust to your timezone
            timestamp = datetime.now(tz).strftime('%Y%m%d_%H%M%S')
        except:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        exp_name = getattr(args, 'exp_name', 'experiment')
        
        wandb.init(
            project="sam-sgp-evaluation",
            name=f"{exp_name}_{timestamp}",
            config={
                "network": args.net,
                "learning_rate": args.lr,
                "image_size": getattr(args, 'image_size', None),
                "batch_size": getattr(args, 'b', None),
                "distributed": is_distributed,
                "world_size": world_size if is_distributed else 1,
                "gpu_device": args.gpu_device if not is_distributed else rank,
                "pretrain": args.pretrain if hasattr(args, 'pretrain') and args.pretrain else "None",
                "evaluation_mode": True,
            },
            tags=["evaluation", args.net, exp_name]
        )
        
        # Log all args to config
        wandb.config.update(vars(args), allow_val_change=True)
        
        # Log system info
        wandb.config.update({
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(GPUdevice) if torch.cuda.is_available() else "CPU",
            "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        })

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net.to(dtype=torch.bfloat16)

    # Load pretrained weights
    if hasattr(args, 'pretrain') and args.pretrain:
        print(f"Loading pretrained weights from: {args.pretrain}")
        weights = torch.load(args.pretrain, map_location=GPUdevice)
        net.load_state_dict(weights["model"])
        
        # Log checkpoint info to wandb
        if wandb_enabled and (not is_distributed or rank == 0):
            checkpoint_info = {
                "checkpoint_path": args.pretrain,
                "checkpoint_loaded": True,
            }
            # Try to extract additional info if available
            if "epoch" in weights:
                checkpoint_info["checkpoint_epoch"] = weights["epoch"]
            if "best_dice" in weights:
                checkpoint_info["checkpoint_best_dice"] = weights["best_dice"]
            if "best_iou" in weights:
                checkpoint_info["checkpoint_best_iou"] = weights["best_iou"]
            
            wandb.config.update(checkpoint_info)
            
            # Log model architecture summary
            total_params = sum(p.numel() for p in net.parameters())
            trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": total_params - trainable_params,
            })
    
    if is_distributed:
        net = DDP(net, device_ids=[rank])
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load data
    print("Loading datasets...")
    nice_train_loader, nice_test_loader = get_dataloader(args)
    
    if wandb_enabled and (not is_distributed or rank == 0):
        wandb.config.update({
            "train_dataset_size": len(nice_train_loader.dataset) if hasattr(nice_train_loader, 'dataset') else "Unknown",
            "test_dataset_size": len(nice_test_loader.dataset) if hasattr(nice_test_loader, 'dataset') else "Unknown",
        })
    
    # Evaluation
    net.eval()
    print("Starting evaluation...")
    eval_start_time = time.time()
    
    loss, iou, dice = function.validation_sam_combined(args, nice_test_loader, 0, net)

    eval_time = time.time() - eval_start_time

    # Aggregate metrics across GPUs in distributed setting
    if is_distributed:
        loss_tensor = torch.tensor(loss, device=GPUdevice)
        iou_tensor = torch.tensor(iou, device=GPUdevice)
        dice_tensor = torch.tensor(dice, device=GPUdevice)
        
        dist.all_reduce(loss_tensor)
        dist.all_reduce(iou_tensor)
        dist.all_reduce(dice_tensor)
        
        loss = (loss_tensor / world_size).item()
        iou = (iou_tensor / world_size).item()
        dice = (dice_tensor / world_size).item()
        
        if rank == 0:
            print(f"[Rank {rank}] Evaluation Results:")
            print(f"  Loss: {loss:.6f}")
            print(f"  IOU: {iou:.6f}")
            print(f"  Dice: {dice:.6f}")
            print(f"  Time: {eval_time:.2f}s")
    else:
        print("Evaluation Results:")
        print(f"  Loss: {loss:.6f}")
        print(f"  IOU: {iou:.6f}")
        print(f"  Dice: {dice:.6f}")
        print(f"  Time: {eval_time:.2f}s")
    
    # Log to wandb - only on rank 0
    if wandb_enabled and (not is_distributed or rank == 0):
        wandb.log({
            'eval/loss': loss,
            'eval/iou': iou,
            'eval/dice': dice,
            'eval/time_seconds': eval_time,
        })
        
        # Create a summary table
        summary_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["Loss", f"{loss:.6f}"],
                ["IOU", f"{iou:.6f}"],
                ["Dice", f"{dice:.6f}"],
                ["Evaluation Time (s)", f"{eval_time:.2f}"],
            ]
        )
        wandb.log({"evaluation_summary": summary_table})
        
        # Update summary with final metrics
        wandb.run.summary["final_loss"] = loss
        wandb.run.summary["final_iou"] = iou
        wandb.run.summary["final_dice"] = dice
        wandb.run.summary["evaluation_time"] = eval_time
        
        # Create a bar chart for metrics
        try:
            wandb.log({
                "metrics_bar_chart": wandb.plot.bar(
                    wandb.Table(
                        columns=["Metric", "Score"],
                        data=[["IOU", iou], ["Dice", dice]]
                    ),
                    "Metric",
                    "Score",
                    title="Evaluation Metrics"
                )
            })
        except Exception as e:
            print(f"Could not create bar chart: {e}")
        
        print("\nResults logged to WandB successfully!")
        print(f"View your run at: {wandb.run.get_url()}")
        
        # Finish the wandb run
        wandb.finish()
            
    if is_distributed:
        cleanup()

def main():
    # Set random seeds for reproducibility
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = cfg.parse_args()
    
    # Check if distributed is actually enabled (not "none" string)
    is_distributed = args.distributed not in [False, 'none', 'None', None]
    wandb_enabled = getattr(args, 'wandb_enabled', False)
    exp_name = getattr(args, 'exp_name', 'experiment')
    
    print("="*60)
    print("SAM Evaluation Script")
    print("="*60)
    print(f"Experiment: {exp_name}")
    print(f"Network: {args.net}")
    print(f"Distributed: {is_distributed}")
    print(f"WandB Enabled: {wandb_enabled}")
    print("="*60)
    
    if is_distributed:
        world_size = torch.cuda.device_count()
        print(f"Running distributed evaluation on {world_size} GPUs")
        mp.spawn(evaluate, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Running single-GPU evaluation")
        evaluate()
    
    print("="*60)
    print("Evaluation completed!")
    print("="*60)

if __name__ == '__main__':
    main()
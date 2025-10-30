import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import cfg
import func_2d.function_3d as function
from conf import settings
from sam_lora_image_encoder import LoRA_Sam
from func_2d.dataset import *
from func_2d.utils import *
from func_3d.dataset import get_dataloader
from evaluate_3d import evaluate_sam


def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)
    torch.cuda.set_device(GPUdevice)

    # Initialize wandb
    wandb.init(
        project="sam-lora-segmentation",  # Change this to your project name
        name=f"{args.net}_{args.exp_name}_{settings.TIME_NOW}",
        config={
            "learning_rate": args.lr,
            "epochs": settings.EPOCH,
            "batch_size": args.b,
            "image_size": args.image_size,
            "network": args.net,
            "optimizer": "Adam",
            "scheduler": "StepLR",
            "lora_rank": 16,
        }
    )
    
    # Update config with all args
    wandb.config.update(args)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    if args.pretrain:
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights, strict=False)

    net = LoRA_Sam(net, 16)
    net.to(device=GPUdevice)
    
    # Log model architecture
    wandb.watch(net, log="all", log_freq=100)
    
    # optimisation
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    '''load pretrained model'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']

        net.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'], strict=False)

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
        
        # Log resume info to wandb
        wandb.config.update({"resumed_from_epoch": start_epoch})
    else:
        start_epoch = 0

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    '''load data'''
    nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = get_dataloader(args)

    '''checkpoint path'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begin training'''
    best_tol = 1e4
    best_dice = 0.0

    for epoch in range(settings.EPOCH):
        nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = get_dataloader(args)
        
        # training
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, net, optimizer, nice_train_loader, nice_support_loader, epoch)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        training_time = time_end - time_start
        
        # Log training metrics
        wandb.log({
            "train/loss": loss,
            "train/epoch": epoch,
            "train/time": training_time,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        print('time_for_training ', training_time)

        # validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH - 1:

            tol, (eiou, edice) = function.validation_sam(args, nice_val_loader, nice_support_loader, epoch, net, None)
            logger.info(f'Val score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')

            # Log validation metrics
            wandb.log({
                "val/total_score": tol,
                "val/iou": eiou,
                "val/dice": edice,
                "val/epoch": epoch
            })

            if args.distributed != 'none':
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            if edice > best_dice:
                best_dice = edice
                best_tol = tol
                is_best = True

                # Log best metrics
                wandb.run.summary["best_dice"] = best_dice
                wandb.run.summary["best_tol"] = best_tol
                wandb.run.summary["best_epoch"] = epoch

                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': args.net,
                    'state_dict': sd,
                    'optimizer': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'best_tol': best_tol,
                    'path_helper': args.path_helper,
                }, is_best, args.path_helper['ckpt_path'], filename="best_dice_checkpoint.pth")
                
                # Save model to wandb
                wandb.save(os.path.join(args.path_helper['ckpt_path'], 'best_dice_checkpoint.pth'))
            else:
                is_best = False

    # Final evaluation on test set
    checkpoint_final = torch.load(os.path.join(args.path_helper['ckpt_path'], 'best_dice_checkpoint.pth'))
    net.load_state_dict(checkpoint_final['state_dict'])
    net.eval()
    tol, (eiou, edice) = function.validation_sam(args, nice_test_loader, nice_support_loader, epoch, net, None)
    logger.info(f'Test score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
    
    # Log test metrics
    wandb.log({
        "test/total_score": tol,
        "test/iou": eiou,
        "test/dice": edice
    })
    
    # Save final results to wandb summary
    wandb.run.summary["final_test_dice"] = edice
    wandb.run.summary["final_test_iou"] = eiou
    wandb.run.summary["final_test_score"] = tol
    
    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
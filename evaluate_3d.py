import os
import time
from tqdm import tqdm
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
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
from sam2_train.build_sam import build_sam2

def main():
    # use bfloat16 for the entire work
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)
    torch.cuda.set_device(GPUdevice)

    # Initialize wandb
    wandb.init(
        project="sam-lora-evaluation",
        name=f"eval_{args.net}_{args.task}_{settings.TIME_NOW}",
        config={
            "network": args.net,
            "image_size": args.image_size,
            "support_size": args.support_size,
            "memory_bank_size": args.memory_bank_size,
            "task": args.task,
            "learning_rate": args.lr,
        }
    )
    wandb.config.update(args)

    sam_net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution=args.distributed)
    net = LoRA_Sam(sam_net, 16)
    net.to(device=GPUdevice)

    '''load data'''
    dataloaders = get_dataloader(args)
    if len(dataloaders) == 4: 
        nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader = dataloaders
    
    else: 
        nice_train_loader, nice_test_loader = dataloaders
    
    '''load pretrained model FIRST'''
    if args.weights != 0:
        print(f'=> resuming from {args.weights}')
        assert os.path.exists(args.weights)
        checkpoint_file = os.path.join(args.weights)
        assert os.path.exists(checkpoint_file)
        loc = 'cuda:{}'.format(args.gpu_device)
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        
        best_dice = checkpoint.get('best_dice', 'N/A')
        checkpoint_epoch = checkpoint.get('epoch', 'N/A')
        print(f'best_dice={best_dice} at epoch {checkpoint_epoch}')
        
        wandb.config.update({
            "loaded_checkpoint": args.weights,
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_best_dice": best_dice
        })
        
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        args.path_helper = checkpoint['path_helper']
        net.to(device=GPUdevice)
        print(f'=> loaded checkpoint {checkpoint_file}')
    else:
        print("Warning: No checkpoint loaded. Running with random initialization.")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    net.eval()
    val_loss, (val_iou, val_dice) = evaluate_sam(args, nice_val_loader, nice_support_loader, "val", net)
    print(f'Validation - Loss: {val_loss:.4f}, IOU: {val_iou:.4f}, DICE: {val_dice:.4f}')
    wandb.log({
        "val/loss": val_loss,
        "val/iou": val_iou,
        "val/dice": val_dice,
    })
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    tol, (eiou, edice) = evaluate_sam(args, nice_test_loader, nice_support_loader, "test", net)
    print(f'Test - Loss: {tol:.4f}, IOU: {eiou:.4f}, DICE: {edice:.4f}')
    wandb.log({
        "test/loss": tol,
        "test/iou": eiou,
        "test/dice": edice,
    })
    
    # Save final summary
    wandb.run.summary["final_test_dice"] = edice
    wandb.run.summary["final_test_iou"] = eiou
    wandb.run.summary["final_test_loss"] = tol
    wandb.run.summary["final_val_dice"] = val_dice
    wandb.run.summary["final_val_iou"] = val_iou
    
    print("\nEvaluation complete! Finishing wandb run...")
    # Finish wandb run
    wandb.finish()
    print("Done!")

def evaluate_sam(args, val_loader, support_loader, split_name, net: nn.Module, clean_dir=True):
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True
    # eval mode
    net.eval()

    GPUdevice = torch.device('cuda',args.gpu_device)
    pos_weight = torch.ones([1]).cuda(device=GPUdevice) * 2
    criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mask_type = torch.float32

    n_val = len(val_loader)
    threshold = (0.5, 0.5, 0.5, 0.5, 0.5)

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    highres_size = args.image_size // 4
    feat_sizes = [(highres_size // (2 ** k), highres_size // (2 ** k)) for k in range(3)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0
    flag = 0
    
    # For wandb visualization - collect sample predictions
    sample_predictions = []
    sample_count = 0
    max_samples = 5  # Number of samples to log to wandb

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, pack in enumerate(val_loader):
            # input image and gt masks
            imgs_input = pack['image'].to(dtype=mask_type, device=GPUdevice)
            masks_input = pack['mask'].to(dtype=mask_type, device=GPUdevice)
            support_data = next(iter(support_loader))
            support_imgs = support_data['image'][0].to(dtype=mask_type, device=GPUdevice)
            support_masks = support_data['mask'][0].to(dtype=mask_type, device=GPUdevice)
            name = pack['image_meta_dict']['filename_or_obj'][0]

            '''generate support embeddings'''
            support_backbone_out = net.sam.forward_image(support_imgs)
            _, support_vision_feats, support_vision_pos_embeds, _ = net.sam._prepare_backbone_features(
                support_backbone_out)
            '''encode support mask into support embeddings'''
            high_res_supportmasks = F.interpolate(support_masks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            supportmem_features, supportmem_pos_enc = net.sam._encode_new_memory(
                current_vision_feats=support_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_supportmasks,
                is_mask_from_pts=True)
            
            supportmem_features = supportmem_features.to(dtype=torch.bfloat16)
            supportmem_features = supportmem_features.to(device=GPUdevice, non_blocking=True)
            supportmem_pos_enc = supportmem_pos_enc[0].to(dtype=torch.bfloat16)
            supportmem_pos_enc = supportmem_pos_enc.to(device=GPUdevice, non_blocking=True)
            support_vision_feats_temp = support_vision_feats[-1].permute(1, 0, 2).reshape(
                support_vision_feats[-1].size(1), -1, 64, 64)
            support_vision_feats_temp = F.normalize(
                support_vision_feats_temp.reshape(support_vision_feats[-1].size(1), -1)).cuda()
            length_3d = imgs_input.size(1)
            memory_bank_feats = []
            memory_bank_pos_enc = []
            final_predictions = []
            
            # Update this part:
            if split_name == "val" and sample_count == 0:  # Only save masks once for val
                raw_imgs = imgs_input[0].permute(2,3,0,1).cpu().numpy()
                raw_masks = masks_input[0].permute(2,3,0,1).cpu().numpy()
                raw_masks[raw_masks>0] = 1
                nii_prediction = nib.Nifti1Image(raw_masks, affine=np.eye(4))
                mask_path = os.path.join('/mnt/disk1/quangminh/SAM2-SGP/result/MSD', 
                                        name[0] + f"_mask_{args.task}_{split_name}.nii.gz")
                nib.save(nii_prediction, mask_path)

            '''test'''
            with torch.no_grad():
                for i in range(length_3d):
                    imgs = imgs_input[:, i, ...]
                    masks = masks_input[:, i, ...]

                    '''Train image encoder'''
                    backbone_out = net.sam.forward_image(imgs)
                    _, vision_feats, vision_pos_embeds, _ = net.sam._prepare_backbone_features(backbone_out)

                    '''Train memory attention to condition on support image'''
                    B = vision_feats[-1].size(1)
                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64).cuda()
                    vision_feats_temp = F.normalize(vision_feats_temp.reshape(B, -1))
                    similarity_scores = F.softmax(torch.mm(support_vision_feats_temp, vision_feats_temp.t()).t(),
                                                  dim=1).cuda()
                    support_samples = torch.multinomial(similarity_scores, num_samples=args.support_size)

                    '''memory attention on support embeddings and input image feats'''
                    support_memory = supportmem_features[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(
                        non_blocking=True)
                    support_memory_pos_enc = supportmem_pos_enc[support_samples].flatten(3).permute(1, 3, 0, 2).cuda(
                        non_blocking=True)

                    '''memory attention on past frames'''
                    if len(memory_bank_feats) == 0:
                        memory = support_memory.reshape(-1, support_memory.size(2), support_memory.size(3))
                        memory_pos_enc = support_memory_pos_enc.reshape(-1, support_memory_pos_enc.size(2),
                                                                        support_memory_pos_enc.size(3))
                    else:
                        L = len(memory_bank_feats)
                        memory_stack_ori = torch.stack(memory_bank_feats, dim=0)
                        memory_pos_stack_ori = torch.stack(memory_bank_pos_enc, dim=0)
                        memory_3d = memory_stack_ori.flatten(3).permute(0, 3, 1, 2)
                        memory_pos_3d = memory_pos_stack_ori.flatten(3).permute(0, 3, 1, 2)
                        memory = torch.cat([support_memory, memory_3d], dim=0)
                        memory_pos_enc = torch.cat([support_memory_pos_enc, memory_pos_3d], dim=0)
                        memory = memory.reshape(-1, memory.size(2), memory.size(3))
                        memory_pos_enc = memory_pos_enc.reshape(-1, memory_pos_enc.size(2), memory_pos_enc.size(3))

                    vision_feats[-1] = net.sam.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos_enc,
                        num_obj_ptr_tokens=0
                    )
                    feats = [feat.cuda(non_blocking=True).permute(1, 2, 0).view(B, -1, *feat_size)
                             for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                    image_embed = feats[-1]
                    high_res_feats = feats[:-1]

                    with torch.no_grad():
                        '''infer from cross attentioned feats and zero prompts'''
                        se, de = net.sam.sam_prompt_encoder(
                            points=None,
                            boxes=None,
                            masks=None,
                            batch_size=B,
                        )
                        y_0 = net.sam.sam_mask_decoder(
                            image_embeddings=image_embed,
                            image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=se,
                            dense_prompt_embeddings=de,
                            multimask_output=False,
                            repeat_image=False,
                            high_res_features=high_res_feats)[0]
                        '''generate pseudo mask'''
                        y_0_pred = F.interpolate(y_0, size=(args.out_size, args.out_size))
                        '''generate box prompt'''
                        box_corordinates = random_box_gai((y_0_pred > 0.5).float()).cuda()
                        se, de = net.sam.sam_prompt_encoder(
                            points=None,
                            boxes=box_corordinates,
                            masks=None,
                            batch_size=B,
                        )

                    '''train mask decoder'''
                    low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam.sam_mask_decoder(
                        image_embeddings=image_embed,
                        image_pe=net.sam.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=se,
                        dense_prompt_embeddings=de,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=high_res_feats
                    )

                    # prediction
                    pred = F.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
                    high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                    mode="bilinear", align_corners=False)

                    # binary mask and calculate loss, iou, dice
                    total_loss += lossfunc(pred, masks)
                    pred = (pred > 0.5).float()
                    temp = eval_seg(pred, masks, threshold)
                    total_eiou += temp[0]
                    total_dice += temp[1]
                    flag = flag + 1
                    final_predictions.append(pred)

                    '''encode new mask and features to memory'''
                    maskmem_features_3d, maskmem_pos_enc_3d = net.sam._encode_new_memory(
                        current_vision_feats=vision_feats,
                        feat_sizes=feat_sizes,
                        pred_masks_high_res=high_res_multimasks,
                        is_mask_from_pts=False)

                    maskmem_features_3d = maskmem_features_3d.to(torch.bfloat16)
                    maskmem_pos_enc_3d = maskmem_pos_enc_3d[0].to(torch.bfloat16)
                    
                    '''store it in memory bank'''
                    if len(memory_bank_feats) < args.memory_bank_size:
                        memory_bank_feats.append(maskmem_features_3d)
                        memory_bank_pos_enc.append(maskmem_pos_enc_3d)
                    else:
                        memory_bank_feats.pop(0)
                        memory_bank_pos_enc.pop(0)
                        memory_bank_feats.append(maskmem_features_3d)
                        memory_bank_pos_enc.append(maskmem_pos_enc_3d)

            # Save predictions
            prediction = torch.cat(final_predictions, dim=0)
            prediction = prediction * 255
            prediction = prediction.squeeze(1).permute(1,2,0).cpu().numpy().astype(np.uint8)
            prediction[prediction>0] = 1
            nii_prediction = nib.Nifti1Image(prediction, affine=np.eye(4))
            pred_path = os.path.join('/mnt/disk1/quangminh/SAM2-SGP/result/MSD', name[0] + "_prediction_"+args.task+"_"+str(epoch)+".nii.gz")
            nib.save(nii_prediction, pred_path)
            
            # Log sample predictions to wandb
            if sample_count < max_samples:
                # Get middle slice for visualization
                mid_slice = prediction.shape[2] // 2
                pred_slice = prediction[:, :, mid_slice]
                
                # Get corresponding ground truth
                gt_masks = masks_input[0].permute(2,3,0,1).cpu().numpy()
                gt_masks[gt_masks>0] = 1
                gt_slice = gt_masks[:, :, mid_slice]
                
                # Get corresponding image
                img_slice = imgs_input[0, mid_slice, 0].cpu().numpy()
                
                # Create wandb Image with overlays
                sample_predictions.append(wandb.Image(
                    img_slice,
                    masks={
                        "predictions": {"mask_data": pred_slice, "class_labels": {1: "predicted"}},
                        "ground_truth": {"mask_data": gt_slice, "class_labels": {1: "ground_truth"}}
                    },
                    caption=f"{name[0]}_slice_{mid_slice}"
                ))
                sample_count += 1

            pbar.update()

    # Log sample predictions to wandb
    if len(sample_predictions) > 0:
        wandb.log({f"predictions/epoch_{epoch}": sample_predictions})
    
    # Calculate and log per-batch metrics
    n_val = n_val * length_3d
    avg_loss = total_loss / flag
    avg_iou = total_eiou / flag
    avg_dice = total_dice / flag
    
    print(f"Evaluation - Slices: {length_3d}, Processed: {flag}, Total: {n_val}")
    
    # Log batch-level metrics
    wandb.log({
        f"eval_epoch_{epoch}/avg_loss": avg_loss,
        f"eval_epoch_{epoch}/avg_iou": avg_iou,
        f"eval_epoch_{epoch}/avg_dice": avg_dice,
        f"eval_epoch_{epoch}/num_slices": length_3d,
        f"eval_epoch_{epoch}/processed_samples": flag,
    })
    
    return avg_loss, tuple([avg_iou, avg_dice])


if __name__ == '__main__':
    main()
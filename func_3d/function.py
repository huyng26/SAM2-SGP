

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm
import wandb
import cfg
from conf import settings
from func_3d.utils import eval_seg, iou_score, CombinedLoss, update_loss, average_loss, update_score, average_score, extract_object, sample_diverse_support, calculate_bounding_box, extract_object_multiple

args = cfg.parse_args()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []



def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image'] #[1,4,3,1024,1024]
            mask_dict = pack['label']
            print(imgs_tensor.shape)
            print(mask_dict.keys())
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0) #[4, 3, 1024, 1024]
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor) #train_init_state
            prompt_frame_id = list(range(0, video_length, prompt_freq)) #prompt id
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list)) #obj_list
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5

            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id]#.to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                net.reset_state(train_state)

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox':
                                bbox = bbox_dict[id][ann_obj_id]
                                _, _, _ = net.train_add_new_bbox(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    bbox=bbox.to(device=GPUdevice),
                                    clear_old_points=False,
                                )
                        except KeyError:
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=id,
                                obj_id=ann_obj_id,
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.vis:
                            os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy())
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        pred_dice += temp[1]

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])

def validation_sam_combined(args, val_loader, epoch, net: nn.Module, clean_dir=True, rank=None):
    if args.distributed:
        net = net.module
        GPUdevice = torch.device('cuda', rank)
    else:
        GPUdevice = torch.device('cuda', args.gpu_device)

    # eval mode
    net.eval()
    n_val = len(val_loader)

    total_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
    dice_score_per_class = {}

    lossfunc = paper_loss

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for packs in val_loader:
            torch.cuda.empty_cache()
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype = torch.float32, device = GPUdevice)
            name = packs["name"][0]
            # Log initial slice stats for validation
            # print(f"[VALIDATION PACK] Name: {name}")
            # print(f"  Query Total Slices: {whole_masks_tensor.shape[0]}, Classes: {torch.unique(whole_masks_tensor)}")
            # print(f"  Support Total Slices: {whole_support_masks_tensor.shape[0]}, Classes: {torch.unique(whole_support_masks_tensor)}")

            obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
            instance_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
            for obj_id in obj_list:
                pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor, \
                                      obj_id=obj_id, video_length=None, num_support=args.num_support)
                if pack is None:
                    print(f"[Validation] [PACK]: No valid for pack for obj_id={obj_id}. Skipping...")
                    # print(f"[DEBUG - QUERY] Slices: {whole_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_masks_tensor)}")
                    # print(f"[DEBUG - SUPPORT] Slices: {whole_support_imgs_tensor.shape[0]}, Unique Classes: {torch.unique(whole_support_masks_tensor)}")
                    continue
                if obj_id not in dice_score_per_class.keys():
                    dice_score_per_class[obj_id] = {"dice_score":0, "num_step": 0} 
                imgs_tensor = pack['image']
                masks_tensor = pack['label']

                # selected_support_frames = torch.randint(0, len(masks_tensor), size=(10,)).tolist()
                # support_imgs_tensor = pack["image"][selected_support_frames]
                # support_masks_tensor = pack["label"][selected_support_frames]

                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                # support_bbox_dict = pack["support_bbox"]
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Query] Warning: Empty image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue  # Skip empty tensors

                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    print(f"VALIDATION: [Support] Warning: Empty support image or mask tensor for obj_id={obj_id} in {name}. Skipping...")
                    continue

                train_state = net.val_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor)
                
                support_pair = []
                filtered_support_pair = []
                for frame_idx in range(support_imgs_tensor.shape[0]):
                    support_image = support_imgs_tensor[frame_idx].permute(1, 2, 0).detach().cpu().numpy()
                    support_label = support_masks_tensor[frame_idx].detach().cpu().numpy()
                    support_pair.append(wandb.Image(support_image, masks={"ground_truth": {"mask_data": support_label}}, caption=f"support {frame_idx}"))
                     # Add to the filtered support pair if the label contains the current obj_id
                    if (support_label == obj_id).any():
                        filtered_support_pair.append(
                            wandb.Image(
                                support_image,
                                masks={"ground_truth": {"mask_data": support_label}},
                                caption=f"Support {frame_idx} (class {obj_id})"
                            )
                        )
                
                if args.wandb_enabled:
                    wandb.log({"test/support set": support_pair})
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for frame_idx in range(support_masks_tensor.shape[0]):
                            mask = support_masks_tensor[frame_idx]
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                mask=mask.to(device=GPUdevice),
                            )

                                # bbox = support_bbox_dict[frame_idx][ann_obj_id]
                                # _, _, _ = net.train_add_new_bbox(
                                #     inference_state=train_state,
                                #     frame_idx=frame_idx,
                                #     obj_id=ann_obj_id,
                                #     bbox=bbox.to(device=GPUdevice),
                                #     clear_old_points=False,
                                # )

                            # else:
                            #     _, _, _ = net.train_add_new_mask(
                            #         inference_state=train_state,
                            #         frame_idx=frame_idx,
                            #         obj_id=ann_obj_id,
                            #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            #     )

                        video_segments = {}  # video_segments contains the per-frame segmentation results
                    
                        for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                            video_segments[out_frame_idx] = {
                                out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                                "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                                for i, out_obj_id in enumerate(out_obj_ids)
                            }
                
                
                # frame_predicted_iou = [(frame_idx, output["iou"]) for frame_idx, object_output in video_segments.items() for obj_id, output in object_output.items()]
                # if imgs_tensor.shape[0] < support_imgs_tensor.shape[0]:
                #     new_support_num_frame = imgs_tensor.shape[0]
                # else:
                #     new_support_num_frame = support_imgs_tensor.shape[0]
                # top10_frame = [frame_idx for frame_idx, iou in sorted(frame_predicted_iou, key=lambda x: x[1], reverse=True)[:new_support_num_frame]]
                # new_support_imgs_tensor = imgs_tensor[top10_frame]
                # new_support_masks_tensor = torch.zeros(new_support_num_frame, 1024, 1024).to(device=GPUdevice)
                # for frame_idx, object_id_output in video_segments.items():
                #     if frame_idx in top10_frame:
                #         for obj_id, output in object_id_output.items():
                #             mask = torch.sigmoid(output["pred_mask"]) >= 0.5
                #             new_support_masks_index = top10_frame.index(frame_idx)
                #             new_support_masks_tensor[new_support_masks_index] = mask
                
                # train_state = net.val_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=new_support_imgs_tensor)

                # with torch.no_grad():
                #     with torch.cuda.amp.autocast():
                #         for frame_idx in range(new_support_masks_tensor.shape[0]):
                #             mask = new_support_masks_tensor[frame_idx]
                #             _, _, _ = net.train_add_new_mask(
                #                 inference_state=train_state,
                #                 frame_idx=frame_idx,
                #                 obj_id=obj_id,
                #                 mask=mask.to(device=GPUdevice),
                #             )

                #             #     bbox = support_bbox_dict[frame_idx][ann_obj_id]
                #             #     _, _, _ = net.train_add_new_bbox(
                #             #         inference_state=train_state,
                #             #         frame_idx=frame_idx,
                #             #         obj_id=ann_obj_id,
                #             #         bbox=bbox.to(device=GPUdevice),
                #             #         clear_old_points=False,
                #             #     )

                #             # else:
                #             #     _, _, _ = net.train_add_new_mask(
                #             #         inference_state=train_state,
                #             #         frame_idx=frame_idx,
                #             #         obj_id=ann_obj_id,
                #             #         mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                #             #     )

                #         video_segments = {}  # video_segments contains the per-frame segmentation results
                    
                #         for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.propagate_in_video(train_state):
                #             video_segments[out_frame_idx] = {
                #                 out_obj_id: {"image_tensor": imgs_tensor[out_frame_idx], "image_label" : masks_tensor[out_frame_idx],
                #                 "pred_mask": out_mask_logits[i], "iou": ious[i], "object_score_logits": object_score_logits[i]}
                #                 for i, out_obj_id in enumerate(out_obj_ids)
                #             }


                # Record the loss in this step
                wandb_result = []
                class_score = {"total_score": 0, "dice_score": 0, "iou_score": 0, "num_step": 0}
                for frame_idx in video_segments.keys():
                    if args.wandb_enabled:
                        whole_gt_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                        whole_pred_mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice)
                        output = video_segments[frame_idx]
                        original_img_wandb = imgs_tensor[frame_idx, 0, :, :].detach().cpu().numpy()

                        pred_mask = torch.where(torch.sigmoid(output[obj_id]["pred_mask"][0, :, :])>=0.5, obj_id, 0)
                        whole_pred_mask += pred_mask

                        gt_mask = output[obj_id]["image_label"]
                        if gt_mask is not None:
                            gt_mask = torch.where(gt_mask==1, obj_id, 0).to(device=GPUdevice)
                            whole_gt_mask += gt_mask

                        whole_pred_mask = whole_pred_mask.detach().cpu().numpy()
                        whole_gt_mask = whole_gt_mask.detach().cpu().numpy()
                        # Normalize original image
                        original_img_wandb = (original_img_wandb - original_img_wandb.min()) / (original_img_wandb.max() - original_img_wandb.min())
                        original_img_wandb = (original_img_wandb * 255).astype(np.uint8)  # Scale to 0–255
                        # Overlap visualization
                        overlap = np.stack([original_img_wandb] * 3, axis=-1)  # Shape: (H, W, 3)
                        overlap[whole_gt_mask == obj_id] = [0, 255, 0]  # Green for ground truth
                        overlap[whole_pred_mask == obj_id] = [255, 0, 0]  # Red for prediction
                        overlap[(whole_gt_mask == obj_id) & (whole_pred_mask == obj_id)] = [255, 255, 0]  # Yellow for overlap

                        wandb_result += [[
                            wandb.Image(original_img_wandb, caption="Image"),
                            wandb.Image(original_img_wandb, masks={"ground_truth": {"mask_data": whole_gt_mask}}, caption="Label"),
                            wandb.Image(original_img_wandb, masks={"predictions": {"mask_data": whole_pred_mask}}, caption="Prediction"),
                            wandb.Image(overlap, caption="Overlap")
                        ] + support_pair]  # Append support_pair directly to the WandB result
                        
                        
                    pred = video_segments[frame_idx][obj_id]["pred_mask"].squeeze(0)
                    mask = video_segments[frame_idx][obj_id]["image_label"]
                    if mask is not None:
                        mask = mask.to(dtype=torch.float32, device=GPUdevice)
                        dice_score, iou_score = eval_seg(pred, mask)
                        update_score(class_score, dice_score, iou_score)
                        class_score["num_step"] += 1

                        dice_score_per_class[obj_id]["dice_score"] += dice_score
                        dice_score_per_class[obj_id]["num_step"] += 1

                    else:
                        mask = torch.zeros_like(pred).to(device=GPUdevice)
                
                if args.wandb_enabled:
                    if len(wandb_result) > 5:
                        sampled_index = np.random.choice(range(len(wandb_result)), size=5)
                    else:
                        sampled_index = range(len(wandb_result))
                    wandb_result = [wandb_inputs for frame_idx, wandb_inputs in enumerate(wandb_result) if frame_idx in sampled_index]
                    wandb_result = [wandb_input for wandb_inputs in wandb_result for wandb_input in wandb_inputs]
                    wandb.log({f"test image/ {int(obj_id)}": wandb_result})

                average_score(class_score)
                update_score(instance_score, class_score["dice_score"], class_score["iou_score"])
                instance_score["num_step"] += 1

            average_score(instance_score)
            print(f"Dice score: {instance_score["dice_score"]} IoU score: {instance_score["iou_score"]}")
            update_score(total_score, instance_score["dice_score"], instance_score["iou_score"])
            total_score["num_step"] += 1
            pbar.update()
        
    average_score(total_score)

    dice_score_per_class = {f"{class_}":dice_score_output["dice_score"]/dice_score_output["num_step"] for class_, dice_score_output in dice_score_per_class.items()}

    if args.wandb_enabled:
        for class_, dice_score in dice_score_per_class.items():
            wandb.log({f"test/Class {class_}": dice_score})

    dice_score_string = ""
    for class_, dice_score in dice_score_per_class.items():
        dice_score_string += f"Class: {class_} Dice Score: {dice_score}\n" 

    print(dice_score_string)

    return total_score["iou_score"], total_score["dice_score"]

def train_sam_combined(args, net: nn.Module, optimizer1, optimizer2, train_loader, epoch):
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    
    GPUdevice = torch.device('cuda', args.gpu_device)
    lossfunc = paper_loss
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='batch') as pbar:
        for packs in train_loader:
            torch.cuda.empty_cache()
            
            whole_imgs_tensor = packs["image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice)
            whole_masks_tensor = packs["label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice)
            whole_support_imgs_tensor = packs["support_image"].squeeze(0).to(dtype=torch.float32, device=GPUdevice)
            whole_support_masks_tensor = packs["support_label"].squeeze(0).to(dtype=torch.float32, device=GPUdevice)
            valid_depth = packs["valid_depth"][0]
            name = packs["name"][0]
            
            obj_list = torch.unique(whole_masks_tensor)[1:].int().tolist()
            
            if len(obj_list) == 0:
                pbar.update()
                continue
            
            instance_loss = 0
            instance_prompt_loss = 0
            instance_non_prompt_loss = 0
            num_objects = 0
            
            for obj_id in obj_list:
                pack = extract_object(whole_imgs_tensor, whole_masks_tensor, whole_support_imgs_tensor, whole_support_masks_tensor,
                                      obj_id=obj_id, video_length=args.video_length, num_support=args.num_support)
                
                if pack is None:
                    continue
                
                imgs_tensor = pack['image']
                masks_tensor = pack['label']
                support_imgs_tensor = pack["support_image"]
                support_masks_tensor = pack["support_label"]
                
                if imgs_tensor.numel() == 0 or masks_tensor.numel() == 0:
                    continue
                
                if support_imgs_tensor.numel() == 0 or support_masks_tensor.numel() == 0:
                    continue
                
                video_length = min(imgs_tensor.shape[0], args.video_length) if args.video_length else imgs_tensor.shape[0]
                prompt_freq = args.prompt_freq
                prompt_frame_id = list(range(0, video_length, prompt_freq))
                
                train_state = net.val_init_state(imgs_tensor=imgs_tensor, masks_tensor=masks_tensor, support_imgs_tensor=support_imgs_tensor)
                
                with torch.cuda.amp.autocast():
                    for frame_idx in range(support_masks_tensor.shape[0]):
                        mask = support_masks_tensor[frame_idx]
                        if (mask == obj_id).any():
                            _, _, _ = net.train_add_new_mask(
                                inference_state=train_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                mask=(mask == obj_id).float().to(device=GPUdevice),
                            )
                    
                    video_segments = {}
                    
                    for out_frame_idx, out_obj_ids, ious, object_score_logits, out_mask_logits in net.train_propagate_in_video(train_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: {"pred_mask": out_mask_logits[i], "iou": ious[i]}
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    
                    obj_loss = 0
                    obj_prompt_loss = 0
                    obj_non_prompt_loss = 0
                    
                    for frame_idx in range(video_length):
                        if frame_idx not in video_segments or obj_id not in video_segments[frame_idx]:
                            continue
                        
                        pred = video_segments[frame_idx][obj_id]["pred_mask"].unsqueeze(0)
                        mask = (masks_tensor[frame_idx] == obj_id).float().unsqueeze(0).unsqueeze(0).to(device=GPUdevice)
                        
                        if mask.sum() == 0:
                            continue
                        
                        frame_loss = lossfunc(pred, mask)
                        obj_loss += frame_loss.item()
                        
                        if frame_idx in prompt_frame_id:
                            obj_prompt_loss += frame_loss
                        else:
                            obj_non_prompt_loss += frame_loss
                    
                    if video_length > 0:
                        obj_loss = obj_loss / video_length
                        if len(prompt_frame_id) > 0:
                            obj_prompt_loss = obj_prompt_loss / len(prompt_frame_id)
                        if len(prompt_frame_id) < video_length:
                            obj_non_prompt_loss = obj_non_prompt_loss / (video_length - len(prompt_frame_id))
                        
                        instance_loss += obj_loss
                        
                        if obj_prompt_loss != 0:
                            if optimizer1 is not None:
                                obj_prompt_loss.backward(retain_graph=True)
                                optimizer1.step()
                                optimizer1.zero_grad()
                            instance_prompt_loss += obj_prompt_loss.item()
                        
                        if obj_non_prompt_loss != 0:
                            if optimizer2 is not None:
                                obj_non_prompt_loss.backward(retain_graph=True)
                                optimizer2.step()
                                optimizer2.zero_grad()
                            instance_non_prompt_loss += obj_non_prompt_loss.item()
                        
                        num_objects += 1
                
                net.reset_state(train_state)
            
            if num_objects > 0:
                epoch_loss += instance_loss / num_objects
                epoch_prompt_loss += instance_prompt_loss / num_objects
                epoch_non_prompt_loss += instance_non_prompt_loss / num_objects
            
            pbar.update()
    
    num_batches = len(train_loader)
    if num_batches == 0:
        return 0, 0, 0
    
    return epoch_loss / num_batches, epoch_prompt_loss / num_batches, epoch_non_prompt_loss / num_batches

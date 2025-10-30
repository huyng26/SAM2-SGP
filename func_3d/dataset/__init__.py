from .btcv import BTCV
from .amos import AMOS
from .pet_tumor import PETCT
from .acdc import ACDC
from .combined import Combined, CombinedTask02Heart
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch

def pad_depth_to(t: torch.Tensor, D_target: int) -> torch.Tensor:
    """
    Zero-pad a tensor along its depth dimension to D_target.
    Supports:
      - [D, C, H, W]  (images)
      - [D, H, W]     (masks)
    Padding is added to the 'end' of depth.
    """
    if t.ndim == 4:  # [D, C, H, W]
        D, C, H, W = t.shape
        if D >= D_target:
            return t
        # move D to last, pad last, move back
        x = t.permute(1, 2, 3, 0)               # [C, H, W, D]
        x = F.pad(x, (0, D_target - D, 0, 0, 0, 0))  # pad depth
        x = x.permute(3, 0, 1, 2)               # [D_target, C, H, W]
        return x

    elif t.ndim == 3:  # [D, H, W]
        D, H, W = t.shape
        if D >= D_target:
            return t
        x = t.permute(1, 2, 0)                  # [H, W, D]
        x = F.pad(x, (0, D_target - D, 0, 0, 0, 0))
        x = x.permute(2, 0, 1)                  # [D_target, H, W]
        return x

    else:
        raise ValueError(f"pad_depth_to: unsupported tensor shape {tuple(t.shape)}")


def custom_collate_fn(batch):
    """
    Collate dict samples with keys:
      - image:          [D,C,H,W]   (tensor)
      - label:          [D,H,W]     (tensor)  # segmentation (no channel)
      - support_image:  [D,C,H,W]   (tensor)
      - support_label:  [D,H,W]     (tensor)
      - name:           str
    Returns padded batched tensors and valid-depth masks to ignore padded slices.
    """
    # Collect fields
    images          = [b["image"] for b in batch]
    labels          = [b["label"] for b in batch]
    support_images  = [b["support_image"] for b in batch]
    support_labels  = [b["support_label"] for b in batch]
    names           = [b["name"] for b in batch]

    # Basic sanity (shapes)
    def _depth(x):
        if x.ndim == 4: return x.shape[0]  # [D,C,H,W]
        if x.ndim == 3: return x.shape[0]  # [D,H,W]
        raise ValueError(f"Unexpected ndim {x.ndim} for shape {tuple(x.shape)}")

    D_img_max     = max(_depth(x) for x in images)
    D_lbl_max     = max(_depth(x) for x in labels)
    D_sup_img_max = max(_depth(x) for x in support_images)
    D_sup_lbl_max = max(_depth(x) for x in support_labels)

    # We'll use a single Dmax to keep all keys aligned slice-wise per sample
    Dmax = max(D_img_max, D_lbl_max, D_sup_img_max, D_sup_lbl_max)

    # valid-depth masks (True for real, False for padded)
    valid_depth         = []
    support_valid_depth = []

    # Pad each list
    images_padded = []
    labels_padded = []
    support_images_padded = []
    support_labels_padded = []

    for img, lbl, simg, slbl in zip(images, labels, support_images, support_labels):
        d_img  = _depth(img)
        d_lbl  = _depth(lbl)
        d_simg = _depth(simg)
        d_slbl = _depth(slbl)

        images_padded.append(pad_depth_to(img,  Dmax))   # -> [Dmax,C,H,W]
        labels_padded.append(pad_depth_to(lbl,  Dmax))   # -> [Dmax,H,W]
        support_images_padded.append(pad_depth_to(simg, Dmax))
        support_labels_padded.append(pad_depth_to(slbl, Dmax))

        # valid slices for primary and support streams
        valid_depth.append(torch.arange(Dmax) < min(d_img, d_lbl))
        support_valid_depth.append(torch.arange(Dmax) < min(d_simg, d_slbl))

    # Stack to batch tensors
    image  = torch.stack(images_padded, dim=0)           # [B,Dmax,C,H,W]
    label  = torch.stack(labels_padded, dim=0)           # [B,Dmax,H,W]
    s_img  = torch.stack(support_images_padded, dim=0)   # [B,Dmax,C,H,W]
    s_lbl  = torch.stack(support_labels_padded, dim=0)   # [B,Dmax,H,W]
    vmask  = torch.stack(valid_depth, dim=0)             # [B,Dmax] (bool)
    svmask = torch.stack(support_valid_depth, dim=0)     # [B,Dmax] (bool)

    return {
        "image": image,
        "label": label,
        "support_image": s_img,
        "support_label": s_lbl,
        "valid_depth": vmask,                # real slices in primary
        "support_valid_depth": svmask,       # real slices in support
        "name": names                        # keep names as list
    }


def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                  prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                 prompt=args.prompt)
        dataset_size = len(amos_train_dataset)
        indices = list(range(dataset_size))
        split_support = 1
        test_dataset_size = len(amos_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5 * test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        train_sampler = SubsetRandomSampler(indices[split_support:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(amos_train_dataset, batch_size=1, sampler=support_sampler,
                                         pin_memory=False)
        nice_train_loader = DataLoader(amos_train_dataset, batch_size=args.b, sampler=train_sampler,
                                       pin_memory=False)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, sampler=test_sampler,
                                      pin_memory=False)
        nice_val_loader = DataLoader(amos_test_dataset, batch_size=1, sampler=val_sampler,
                                     pin_memory=False)
        '''end'''
    elif args.dataset == 'petct':
        petct_train_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                  prompt=args.prompt)
        petct_test_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                 prompt=args.prompt)
        dataset_size = len(petct_train_dataset)
        indices = list(range(dataset_size))
        split_support = 1
        test_dataset_size = len(petct_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5*test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        train_sampler = SubsetRandomSampler(indices[split_support:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(petct_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8, pin_memory=False)
        nice_train_loader = DataLoader(petct_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=False)
        nice_test_loader = DataLoader(petct_test_dataset, batch_size=1, sampler=test_sampler, num_workers=1, pin_memory=False)
        nice_val_loader = DataLoader(petct_test_dataset, batch_size=1, sampler=val_sampler, num_workers=1,
                                     pin_memory=False)
    elif args.dataset == 'petct_distributed':
        petct_train_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                  prompt=args.prompt)
        petct_test_dataset = PETCT(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                 prompt=args.prompt)
        dataset_size = len(petct_train_dataset)
        indices = list(range(dataset_size)) # train indice
        split_support = 1 # support size
        test_dataset_size = len(petct_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5*test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        #train_sampler = DistributedSampler(torch.utils.data.Subset(petct_train_dataset, indices[split_support:]))
        train_sampler = DistributedSampler(dataset=petct_train_dataset, shuffle=True)
        train_sampler.set_epoch(args.epoch)
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        #test_sampler = DistributedSampler(torch.utils.data.Subset(petct_test_dataset, indices[:split_val]))
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(petct_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8, pin_memory=False)
        nice_train_loader = DataLoader(petct_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8, pin_memory=False)
        nice_test_loader = DataLoader(petct_test_dataset, batch_size=args.b, sampler=test_sampler, num_workers=8, pin_memory=False)
        nice_val_loader = DataLoader(petct_test_dataset, batch_size=args.b, sampler=val_sampler, num_workers=8,
                                     pin_memory=False)

    elif args.dataset == 'acdc':
        acdc_train_dataset = ACDC(args, args.data_path, transform=None, transform_msk=None, mode='Training',
                                    prompt=args.prompt)
        acdc_test_dataset = ACDC(args, args.data_path, transform=None, transform_msk=None, mode='Testing',
                                   prompt=args.prompt)
        dataset_size = len(acdc_train_dataset)
        indices = list(range(dataset_size))
        split_support = 5
        test_dataset_size = len(acdc_test_dataset)
        indices_test = list(range(test_dataset_size))
        split_val = int(np.floor(0.5 * test_dataset_size))
        np.random.shuffle(indices)
        np.random.shuffle(indices_test)
        train_sampler = SubsetRandomSampler(indices[split_support:])
        support_sampler = SubsetRandomSampler(indices[:split_support])
        val_sampler = SubsetRandomSampler(indices_test[split_val:])
        test_sampler = SubsetRandomSampler(indices_test[:split_val])
        nice_support_loader = DataLoader(acdc_train_dataset, batch_size=1, sampler=support_sampler, num_workers=8,
                                         pin_memory=False)
        nice_train_loader = DataLoader(acdc_train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=8,
                                       pin_memory=False)
        nice_test_loader = DataLoader(acdc_test_dataset, batch_size=1, sampler=test_sampler, num_workers=8,
                                      pin_memory=False)
        nice_val_loader = DataLoader(acdc_test_dataset, batch_size=1, sampler=val_sampler, num_workers=8,
                                     pin_memory=False)
    elif args.dataset == 'combined':
        combined_train_dataset = Combined(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        combined_test_dataset = Combined(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)
        
        if args.distributed:
            train_sampler = DistributedSampler(combined_train_dataset, num_replicas=world_size, rank=rank)
            test_sampler = DistributedSampler(combined_test_dataset, num_replicas=world_size, rank=rank)

            nice_train_loader = DataLoader(combined_train_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler, collate_fn=custom_collate_fn)
            nice_test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=test_sampler, collate_fn=custom_collate_fn)
        else:
            nice_train_loader = DataLoader(combined_train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
            nice_test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
        '''end'''
    else:
        print("the dataset is not supported now!!!")
        return nice_train_loader, nice_test_loader
    return nice_train_loader, nice_test_loader, nice_support_loader, nice_val_loader

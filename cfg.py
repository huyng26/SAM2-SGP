import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-task', type=str, default='left+kidney', help='organs name for segmentation')
    parser.add_argument('-vis', type=int, default=10, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=144, help='image_size')
    parser.add_argument('-out_size', type=int, default=144, help='output_size')
    parser.add_argument('-support_size', type=int, default=4, help='image_size')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-disted', default=False, type=bool, help='whether use multi GPU')
    parser.add_argument('-local_rank',  type=int, help='local rank of gpu for dist training')
    parser.add_argument('-device', help='gpu device for dist training')
    parser.add_argument('-dist', default=False, type=bool, help='whether use multi GPU')
    parser.add_argument('-dataset', default='msd' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=8, help='sam checkpoint address')
    parser.add_argument('-epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument(
    '-data_path',
    type=str,
    default='/mnt/disk1/aiotlab/huync/AAAI-MedSam2/Combined_Dataset',
    help='The path of segmentation data')
    opt = parser.parse_args()

    return opt

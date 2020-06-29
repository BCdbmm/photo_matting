# _*_ coding: utf-8 _*_
# @Time : 2020/6/29 下午3:21
# @Author : sunyulong
# @File : train_tnet.py

from models.model import *
from data.datasets import *
import os

init_lr=1e-2
end_lr=1e-6

def config_learning_rate(optimizer, decay=0.9):
    lr =  init_lr* decay
    if lr < end_lr:
        return end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train():
    batch_size=32
    annotation_path='../data/train.txt'
    num_steps_to_save_checkpoint = 128000 // batch_size
    dataset=MattingDataset()
    root_dir=None
    model_dir='../saves'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    matting_dataset = dataset.MattingDataset(annotation_path=annotation_path,
                                             root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(matting_dataset,
                                               batch_size=batch_size,
                                               num_workers=2,
                                               shuffle=True,
                                               drop_last=True)
    tnet=T_mv2_unet()

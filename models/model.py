# _*_ coding: utf-8 _*_
# @Time : 2020/6/20 上午6:28
# @Author : sunyulong
# @File : model.py
import torch
import torch.nn as nn
import torchvision

VGG16_BN_CONFIGS = {
    '13conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
         'M', 512, 512, 512],
    '10conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
    }

def make_layer(cfg,bn=True):
    layers=[]
    inchannel=3
    for v in cfg:
        if v=='M':
            layers+=[nn.MaxPool2d(2,2,return_indices=True)]
        else:
            conv=nn.Conv2d(inchannel,v,3,stride=1,padding=1)
            if bn:
                layers+=[conv,nn.BatchNorm2d(v),nn.ReLU()]
            else:
                layers+=[conv,nn.ReLU()]
            inchannel=v
    return nn.Sequential(*layers)

class FeatureMapExtract(nn.Module):
    def __init__(self,cfg=None,bn=True):
        super(FeatureMapExtract, self).__init__()
        if cfg is None:
            self.cfg=VGG16_BN_CONFIGS['10conv']
        else:
            self.cfg=cfg
        self.bn=bn
        self.features=make_layer(self.cfg,self.bn)
        self._indices=[]
        self._pre_pool_shape=[]
    def forward(self,x):
        for layer in self.features:
            if isinstance(layer,nn.modules.pooling.MaxPool2d):
                self._pre_pool_shape.append(x.size())
                x,indices=layer(x)
                self._indices.append(indices)
            else:
                x=layer(x)
        return x

def featureExtract(cfg=None,pre_trained=True):
    fte=FeatureMapExtract(cfg,bn=True)
    if pre_trained:
        state_dict=torch.load('/home/disk1/all_scripts/photo_matting/models/vgg16_bn-6c64b313.pth')
        keys=[k for k in state_dict]
        new_state=fte.state_dict()
        new_state={k:v for k,v in state_dict.items() if k in new_state}
        fte.load_state_dict(new_state)
    return fte

class Mnet(nn.Module):
    def __init__(self,fte):
        super(Mnet, self).__init__()
        self._head_conv=nn.Conv2d(4,3,5,padding=2)
        self.encode=fte
        self.encode_cfg=self.encode.cfg
        self.decode=self.decode_layer()
        self.final_conv=nn.Conv2d(self.encode_cfg[0],1,5,padding=2)
        self.sig=nn.Sigmoid()
    def forward(self,x):
        x=self._head_conv(x)
        x=self.encode(x)
        indices=self.encode._indices[::-1]
        upshape=self.encode._pre_pool_shape[::-1]
        index=0
        for layer in self.decode:
            if isinstance(layer,nn.modules.pooling.MaxUnpool2d):
                # print(x.shape,indices[index].shape)
                x=layer(x,indices[index],output_size=upshape[index])
                index+=1
            else:
                x=layer(x)
        x=self.final_conv(x)
        x=self.sig(x)
        return x

    def decode_layer(self):
        layers=[]
        strides=[]
        channels=[]
        inchannel=self.encode_cfg[-1]
        cfg_reversed=self.encode_cfg[::-1]
        for i,v in enumerate(cfg_reversed):
            if v=='M':
                strides.append(2)
                channels.append(cfg_reversed[i+1])
        channels.append(channels[-1])
        # print(channels)
        for stride,channel in zip(strides,channels[1:]):
            if stride==2:
                layers+=[nn.MaxUnpool2d(2,2)]
            layers+=[nn.Conv2d(inchannel,channel,5,padding=2),nn.BatchNorm2d(channel),nn.ReLU()]
            inchannel=channel
        return nn.Sequential(*layers)


def loss(alphas_pred, alphas_gt, images=None, epsilon=1e-12):
    losses = torch.sqrt(
        torch.mul(alphas_pred - alphas_gt, alphas_pred - alphas_gt) +
        epsilon)
    loss = torch.mean(losses)
    if images is not None:
        images_fg_gt = torch.mul(images, alphas_gt)
        images_fg_pred = torch.mul(images, alphas_pred)
        images_fg_error = images_fg_pred - images_fg_gt
        losses_image = torch.sqrt(
            torch.mul(images_fg_error, images_fg_error) + epsilon)
        loss += torch.mean(losses_image)
    return loss
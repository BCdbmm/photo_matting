# _*_ coding: utf-8 _*_
# @Time : 2020/6/20 上午6:28
# @Author : sunyulong
# @File : model.py
import torch
import torch.nn as nn
import torchvision
from torch.utils.checkpoint import checkpoint

import torch
import torch.nn as nn
import torchvision as tv

VGG16_BN_MODEL_URL = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'

VGG16_BN_CONFIGS = {
    '13conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
         'M', 512, 512, 512],
    '10conv':
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
}


def make_layers(cfg, batch_norm=False):
    """Copy from: torchvision/models/vgg.

    Changs retrue_indices in MaxPool2d from False to True.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2,
                                          return_indices=True)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v),
                           torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)


class VGGFeatureExtractor(torch.nn.Module):
    """Feature extractor by VGG network."""

    def __init__(self, config=None, batch_norm=True):
        """Constructor.

        Args:
            config: The convolutional architecture of VGG network.
            batch_norm: A boolean indicating whether the architecture
                include Batch Normalization layers or not.
        """
        super(VGGFeatureExtractor, self).__init__()
        self._config = config
        if self._config is None:
            self._config = VGG16_BN_CONFIGS.get('10conv')
        self.features = make_layers(self._config, batch_norm=batch_norm)
        self._indices = None
        self._pre_pool_shapes = None

    def forward(self, x):
        self._indices = []
        self._pre_pool_shapes = []
        for layer in self.features:
            if isinstance(layer, torch.nn.modules.pooling.MaxPool2d):
                self._pre_pool_shapes.append(x.size())
                x, indices = layer(x)
                self._indices.append(indices)
            else:
                x = layer(x)
        return x


def vgg16_bn_feature_extractor(config=None, pretrained=True, progress=True):
    model = VGGFeatureExtractor(config, batch_norm=True)
    if pretrained:
        state_dict = tv.models.utils.load_state_dict_from_url(
            VGG16_BN_MODEL_URL, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


class DIM(torch.nn.Module):
    """Deep Image Matting."""

    def __init__(self, feature_extractor):
        """Constructor.

        Args:
            feature_extractor: Feature extractor, such as VGGFeatureExtractor.
        """
        super(DIM, self).__init__()
        # Head convolution layer, number of channels: 4 -> 3
        self._head_conv = torch.nn.Conv2d(in_channels=4, out_channels=3,
                                          kernel_size=5, padding=2)
        # Encoder
        self._feature_extractor = feature_extractor
        self._feature_extract_config = self._feature_extractor._config
        # Decoder
        self._decode_layers = self.decode_layers()
        # Prediction
        self._final_conv = torch.nn.Conv2d(self._feature_extract_config[0], 1,
                                           kernel_size=5, padding=2)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self._head_conv(x)
        x = self._feature_extractor(x)
        indices = self._feature_extractor._indices[::-1]
        unpool_shapes = self._feature_extractor._pre_pool_shapes[::-1]
        index = 0
        for layer in self._decode_layers:
            if isinstance(layer, torch.nn.modules.pooling.MaxUnpool2d):
                x = layer(x, indices[index], output_size=unpool_shapes[index])
                index += 1
            else:
                x = layer(x)
        x = self._final_conv(x)
        x = self._sigmoid(x)
        return x

    def decode_layers(self):
        layers = []
        strides = [1]
        channels = []
        config_reversed = self._feature_extract_config[::-1]
        for i, v in enumerate(config_reversed):
            if v == 'M':
                strides.append(2)
                channels.append(config_reversed[i + 1])
        channels.append(channels[-1])
        in_channels = self._feature_extract_config[-1]
        for stride, out_channels in zip(strides, channels):
            if stride == 2:
                layers += [torch.nn.MaxUnpool2d(kernel_size=2, stride=2)]
            layers += [torch.nn.Conv2d(in_channels, out_channels,
                                       kernel_size=5, padding=2),
                       torch.nn.BatchNorm2d(num_features=out_channels),
                       torch.nn.ReLU(inplace=True)]
            in_channels = out_channels
        return torch.nn.Sequential(*layers)


def loss(alphas_pred, alphas_gt, masks=None, images=None, epsilon=1e-12):
    diff = alphas_pred - alphas_gt
    # diff = diff * masks
    # num_unkowns = torch.sum(masks) + epsilon
    losses = torch.sqrt(torch.mul(diff, diff) + epsilon)
    # loss = torch.sum(losses) / num_unkowns
    loss = torch.mean(losses)
    if images is not None:
        images_fg_gt = torch.mul(images, alphas_gt)
        images_fg_pred = torch.mul(images, alphas_pred)
        images_fg_diff = images_fg_pred - images_fg_gt
        images_fg_diff = images_fg_diff * masks
        losses_image = torch.sqrt(
            torch.mul(images_fg_diff, images_fg_diff) + epsilon)
        loss += torch.mean(losses_image)
        # loss += torch.sum(losses_image) / num_unkowns
    return loss


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_conn = (self.stride == 1 and inp == oup)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expansion, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride=self.stride, padding=1, groups=inp * expansion,
                      bias=False),
            nn.BatchNorm2d(inp*expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp*expansion,oup,1,1,0,bias=False),
            nn.BatchNorm2d(oup)
        )
    def forward(self,x):
        if self.use_res_conn:
            return x+self.conv(x)
        else:
            return self.conv(x)
class mobilenet_v2(nn.Module):
    def __init__(self,nInputChannels=3):
        super(mobilenet_v2, self).__init__()
        self.head_conv=nn.Sequential(
            nn.Conv2d(nInputChannels,32,3,1,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block_1=InvertedResidual(32,16,1,1)
        self.block_2=nn.Sequential(
            InvertedResidual(16,24,2,6),
            InvertedResidual(24,24,1,6)
        )
        self.block_3=nn.Sequential(
            InvertedResidual(24,32,2,6),
            InvertedResidual(32,32,1,6),
            InvertedResidual(32, 32, 1, 6)
        )
        self.block_4=nn.Sequential(
            InvertedResidual(32,64,2,6),
            InvertedResidual(64,64,1,6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6)
        )
        self.block_5=nn.Sequential(
            InvertedResidual(64,96,1,6),
            InvertedResidual(96,96,1,6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.block_6=nn.Sequential(
            InvertedResidual(96,160,2,6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6)
        )
        self.block_7=InvertedResidual(160,320,1,6)
    def forward(self,x):
        x=self.head_conv(x)
        s1=self.block_1(x)
        s2=self.block_2(s1)
        s3=self.block_3(s2)
        s4=self.block_4(s3)
        s4=self.block_5(s4)
        s5=self.block_6(s4)
        s5=self.block_7(s5)
        return s1,s2,s3,s4,s5

class T_mv2_unet(nn.Module):
    def __init__(self,cls_num=3):
        super(T_mv2_unet, self).__init__()
        # encode
        self.encode=mobilenet_v2()
        # decode
        self.s5_up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(320,96,3,1,1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.s4_fusion=nn.Sequential(
            nn.Conv2d(96,96,3,1,1),
            nn.BatchNorm2d(96)
        )
        self.s4_up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(96,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.s3_fusion=nn.Sequential(
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32)
        )
        self.s3_up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(32,24,3,1,1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.s2_fusion=nn.Sequential(
            nn.Conv2d(24,24,3,1,1),
            nn.BatchNorm2d(24)
        )
        self.s2_up_conv=nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(24,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.s1_fusion=nn.Sequential(
            nn.Conv2d(16,16,3,1,1),
            nn.BatchNorm2d(16)
        )
        self.last_conv=nn.Conv2d(16,cls_num,3,1,1)
        self.last_up=nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self,x):
        s1,s2,s3,s4,s5=self.encode(x)
        s4_=self.s5_up_conv(s5)
        s4_=s4_+s4
        s4=self.s4_fusion(s4_)
        s3_=self.s4_up_conv(s4)
        s3_=s3_+s3
        s3=self.s3_fusion(s3_)
        s2_=self.s3_up_conv(s3)
        s2_=s2_+s2
        s2=self.s2_fusion(s2_)
        s1_=self.s2_up_conv(s2)
        s1_=s1_+s1
        s1=self.s1_fusion(s1_)
        out=self.last_conv(s1)
        return out
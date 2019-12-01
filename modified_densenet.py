import re
import os, glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
import time

import pdb

__all__ = ['DenseNet', 'densenet121',
           'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}

writer = SummaryWriter()
def densenet121(type, pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if (type == "va-densenet"):
        model = DenseNetVa(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "reva-densenet"):
        model = DenseNetReva(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "fp-densenet"):
        model = DenseNetFP(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "start-densenet"):
        model = DenseNetStart(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "every-densenet"):
        model = DenseNetEvery(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "sedensenet"):
        model = SEDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "triplelossdensenet"):
        model = TripleLossDenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    elif (type == "aux-densenet"):
        model = DenseNetAux(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)

    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        origin_model = model_zoo.load_url(model_urls['densenet121'])
        for key in list(origin_model.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                origin_model[new_key[9:]] = origin_model[key]
                del origin_model[key]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        origin_model = {k: v for k, v in origin_model.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(origin_model)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _SEBlock(nn.Module):

    def __init__(self, in_ch, r=16):
        super(_SEBlock, self).__init__()

        self.se_linear1 = nn.Linear(in_ch, in_ch//r)
        self.se_linear2 = nn.Linear(in_ch//r, in_ch)

    def forward(self,x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.se_linear1(x), inplace=True)
        x = self.se_linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
  
        x = torch.mul(input_x, x)
        return x


def interpolate(x, multiplier=2, fixed_size=0, divider=2, absolute_channel = 0, mode='nearest'):
    if mode == 'bilinear':
        return F.interpolate(x.view(x.size()[0], x.size()[1], x.size()[2], x.size()[3]),
                                size=(fixed_size if fixed_size != 0 else x.size()[2] * multiplier, 
                                        fixed_size if fixed_size != 0 else x.size()[3] * multiplier),
                                mode=mode)

    return F.interpolate(x.view(1, x.size()[0], x.size()[1], x.size()[2], x.size()[3]),
                            size=(x.size()[1] // divider if absolute_channel == 0 else absolute_channel,
                                    fixed_size if fixed_size != 0 else x.size()[2] * multiplier, 
                                    fixed_size if fixed_size != 0 else x.size()[3] * multiplier),
                            mode=mode)[0]


def print_attention(org_image, input_tensor, gt, timestamp, custom_label):
    # gt = ground truth label
    attention_res = interpolate(input_tensor, fixed_size=224, absolute_channel=1, mode='bilinear')
    for i in range(len(gt)):
        if not os.path.exists(f'attention_image/{gt[i]}_{i}_{timestamp}'):
            os.makedirs(f'attention_image/{gt[i]}_{i}_{timestamp}')

        np.save(f'attention_image/{gt[i]}_{i}_{timestamp}/org_{timestamp}', org_image[i].cpu().data.numpy())
        np.save(f'attention_image/{gt[i]}_{i}_{timestamp}/{timestamp}_{custom_label}', attention_res[i].cpu().data.numpy())


# VA Densenet
# ================================================================
class DenseNetVa(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetVa, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)
 
        # Vanilla Linear Visual attention layer
        # self.valinear = nn.Linear(1024 * 7 * 7, 49)
        self.valinear = nn.Conv2d(1024, 1, 3, 1, 1)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):

        # =============================================================
        # Phase 1 Densenet
        x = self.denseblock1(self.features(x))
        x = self.denseblock2(self.transition1(x))
        x = self.denseblock3(self.transition2(x))
        x = self.denseblock4(self.transition3(x))

        # Vanilla relu visual attention
        va = self.valinear(x)
        x = x + va.view(x.size()[0], 1, x.size()[2], x.size()[3])

        # if epoch != -1:
        #     writer.add_image('Image', vutils.make_grid(x, normalize=True, scale_each=True), epoch)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x


# REVA Densenet
# ================================================================
class DenseNetReva(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetReva, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)
 
        # Feature pyramid
        self.conv2d1x1fp4 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp3 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2d1x1fp1 = nn.Conv2d(256, 256, 1, stride=1, padding=0) 

        # Transconv upsampling
        # input size 7
        self.transconv1 = nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1) # output 14x14
        self.transconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1) # output 28x28
        self.transconv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # output 56x56
        self.transconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # output 112x112
        self.transconv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # output 224x224
        self.transconv6 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1) # output 224x224

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):

        # =============================================================
        # Phase 1 Densenet
        features = self.denseblock1(self.features(x))
        features = self.denseblock2(self.transition1(features))
        features = self.denseblock3(self.transition2(features))
        features = self.denseblock4(self.transition3(features))

        features = self.transconv6(self.transconv5(self.transconv4(self.transconv3(self.transconv2(self.transconv1(features))))))

        x = x + features

        features = self.denseblock1(self.features(x))
        features = self.denseblock2(self.transition1(features))
        features = self.denseblock3(self.transition2(features))
        features = self.denseblock4(self.transition3(features))

        # if epoch != -1:
        #     writer.add_image('Image', vutils.make_grid(x, normalize=True, scale_each=True), epoch)

        x = F.relu(features, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x
        

# FP Densenet
# ================================================================
class DenseNetFP(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetFP, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)
 
        # Feature pyramid
        self.conv2d1x1fp3 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2d1x1fp1 = nn.Conv2d(256, 256, 1, stride=1, padding=0) 

        # Transconv upsampling
        # input size 7
        self.transconv1 = nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1) # output 14x14
        self.transconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1) # output 28x28
        self.transconv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # output 56x56
        self.transconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # output 112x112
        self.transconv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # output 224x224
        self.transconv6 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1) # output 224x224

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):

        # =============================================================
        # Phase 1 Densenet
        f1 = self.denseblock1(self.features(x))
        f2 = self.denseblock2(self.transition1(f1))
        f3 = self.denseblock3(self.transition2(f2))
        f4 = self.denseblock4(self.transition3(f3))

        # =============================================================
        # Phase 2 Feature Pyramid
        # fp3 = interpolate(f4, divider=1) + self.conv2d1x1fp3(f3) # output 1024, 14, 14
        # fp2 = interpolate(f3) + self.conv2d1x1fp2(f2) # output 512, 28, 28
        # fp1 = interpolate(f2) + self.conv2d1x1fp1(f1) # output 256, 56, 56
        # fp1 = interpolate(fp1) # output 128, 112, 112
        # fp1 = interpolate(fp1) # output 64, 224, 224
        # fp1 = interpolate(fp1, multiplier = 1, absolute_channel = 3) # output 3, 224, 224

        fp3 = self.transconv1(f4) + self.conv2d1x1fp3(f3)
        fp2 = self.transconv2(fp3) + self.conv2d1x1fp2(f2)
        fp1 = self.transconv3(fp2) + self.conv2d1x1fp1(f1)
        fp1 = self.transconv6(self.transconv5(self.transconv4(fp1)))

        x = x + fp1

        # =============================================================
        # Phase 3 normal Densenet Sequence
        x = self.features(x)
        x = self.denseblock1(x)
        x = self.denseblock2(self.transition1(x))
        x = self.denseblock3(self.transition2(x))
        x = self.denseblock4(self.transition3(x))
        x = self.batchNorm5(x)

        # if epoch != -1:
        #     writer.add_image('Image', vutils.make_grid(x, normalize=True, scale_each=True), epoch)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x
        

# Start Densenet
# ================================================================
class DenseNetStart(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetStart, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)

        # Start VA
        self.startconv2d = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):
        attention = self.startconv2d(x)
        x = x + attention 

        x = self.denseblock1(self.features(x))
        x = self.denseblock2(self.transition1(x))
        x = self.denseblock3(self.transition2(x))
        x = self.denseblock4(self.transition3(x))

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x


# Every Densenet
# ================================================================
class DenseNetEvery(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetEvery, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)

        # Every VA
        self.everyconv2dblock1 = nn.Conv2d(2816, 1024, kernel_size=1, stride=1, padding=0)
        self.everyconv2dblock256 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock512 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock1024_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock1024_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, gt = 0):
        # =============================================================
        # Phase 1 Densenet
        current_timestamp = time.time()

        db1 = self.denseblock1(self.features(x))
        attention1 = F.relu(self.everyconv2dblock256(db1))
        # print_attention(x, attention1, gt, current_timestamp, custom_label='256')
        db1 = attention1 + db1

        db2 = self.denseblock2(self.transition1(db1))
        attention2 = F.relu(self.everyconv2dblock512(db2))
        # print_attention(x, attention2, gt, current_timestamp, custom_label='512')
        db2 = attention2 + db2

        db3 = self.denseblock3(self.transition2(db2))
        attention3 = F.relu(self.everyconv2dblock1024_1(db3))
        # print_attention(x, attention3, gt, current_timestamp, custom_label='1024a')
        db3 = attention3 + db3

        db4 = self.denseblock4(self.transition3(db3))
        attention4 = F.relu(self.everyconv2dblock1024_2(db4))
        # print_attention(x, attention4, gt, current_timestamp, custom_label='1024b')
        db4 = attention4 + db4

        db4 = F.relu(db4, inplace=True)
        db4 = F.avg_pool2d(db4, kernel_size=7, stride=1).view(x.size(0), -1)

        db4 = self.classifier(db4)
        return db4
        

# SE Densenet
# ================================================================
class SEDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(SEDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.seblock1 = _SEBlock(in_ch=num_features, r = 16)
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.seblock2 = _SEBlock(in_ch=num_features, r = 16)
        
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.seblock3 = _SEBlock(in_ch=num_features, r = 16)
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        
        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):

        # =============================================================
        # Phase 1 Densenet
        x = self.seblock1(self.transition1(self.denseblock1(self.features(x))))
        x = self.seblock2(self.transition2(self.denseblock2(x)))
        x = self.seblock3(self.transition3(self.denseblock3(x)))
        x = self.denseblock4(x)

        # if epoch != -1:
        #     writer.add_image('Image', vutils.make_grid(x, normalize=True, scale_each=True), epoch)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7, stride=1).view(x.size(0), -1)

        x = self.classifier(x)
        return x


# TripleLossDenseNet
# ================================================================
class TripleLossDenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(TripleLossDenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)
 
        # Feature pyramid
        self.conv2d1x1fp3 = nn.Conv2d(1024, 1024, 1, stride=1, padding=0)
        self.conv2d1x1fp2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2d1x1fp1 = nn.Conv2d(256, 256, 1, stride=1, padding=0) 

        self.conv2dlastblock1 = nn.Conv2d(256, 256, 3, stride=1, padding=1) 
        self.conv2dlastblock2 = nn.Conv2d(512, 512, 3, stride=1, padding=1) 
        self.conv2dlastblock3 = nn.Conv2d(1024, 1024, 3, stride=1, padding=1) 

        # Transconv upsampling
        # input size 7
        self.transconv1 = nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1) # output 14x14
        self.transconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1) # output 28x28
        self.transconv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # output 56x56
        self.transconv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # output 112x112
        self.transconv5 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # output 224x224
        self.transconv6 = nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1) # output 224x224

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch = -1):

        # =============================================================
        # Phase 1 Densenet
        f1 = self.denseblock1(self.features(x))
        f2 = self.denseblock2(self.transition1(f1))
        f3 = self.denseblock3(self.transition2(f2))
        f4 = self.denseblock4(self.transition3(f3))
        result_1 = self.batchNorm5(f4)

        result_1 = F.relu(result_1, inplace=True)
        result_1 = F.avg_pool2d(result_1, kernel_size=7, stride=1).view(result_1.size(0), -1)

        result_1 = self.classifier(result_1)

        # =============================================================
        # Phase 2 Feature Pyramid
        fp3 = self.transconv1(f4) + self.conv2d1x1fp3(f3)
        fp2 = self.transconv2(fp3) + self.conv2d1x1fp2(f2)
        fp1 = self.transconv3(fp2) + self.conv2d1x1fp1(f1)
        fpimage = self.transconv6(self.transconv5(self.transconv4(fp1)))

        x = x + fpimage
        # =============================================================
        # Phase 3 Second Densenet
        fd1 = self.features(x)
        fd1 = self.denseblock1(fd1) + self.conv2dlastblock1(fp1)
        fd2 = self.denseblock2(self.transition1(fd1)) + self.conv2dlastblock2(fp2)
        fd3 = self.denseblock3(self.transition2(fd2)) + self.conv2dlastblock3(fp3)
        fd4 = self.denseblock4(self.transition3(fd3))
        result_2 = self.batchNorm5(fd4)

        result_2 = F.relu(result_2, inplace=True)
        result_2 = F.avg_pool2d(result_2, kernel_size=7, stride=1).view(x.size(0), -1)

        result_2 = self.classifier(result_2)

        # return [result_1, result_2, (result_1 + result_2) / 2]
        return (result_1 + result_2) / 2


# Aux Densenet
# ================================================================
class DenseNetAux(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNetAux, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features

        # Block 1
        num_layers = 6
        self.denseblock1 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 2
        num_layers = 12
        self.denseblock2 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 
        # Block 3
        num_layers = 24
        self.denseblock3 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate
        self.transition3 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        
        # Block 4
        num_layers = 16
        self.denseblock4 = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        # BatchNorm5 
        self.batchNorm5 = nn.BatchNorm2d(num_features)

        # Every VA
        self.everyconv2dblock1 = nn.Conv2d(2816, 1024, kernel_size=1, stride=1, padding=0)
        self.everyconv2dblock256 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock512 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock1024_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock1024_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.classifier1 = nn.Linear(256, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(1024, num_classes)

        # Softmax Sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, gt = 0):
        # =============================================================
        # Phase 1 Densenet
        current_timestamp = time.time()

        db1 = self.denseblock1(self.features(x))
        attention1 = self.sigmoid(self.everyconv2dblock256(db1))
        # print_attention(x, attention1, gt, current_timestamp, custom_label='256')
        att1 = attention1 * db1
        att1 = F.avg_pool2d(att1, kernel_size=56, stride=1).view(x.size(0), -1)
        att1 = self.classifier1(att1)

        db2 = self.denseblock2(self.transition1(db1))
        attention2 = self.sigmoid(self.everyconv2dblock512(db2))
        # print_attention(x, attention2, gt, current_timestamp, custom_label='512')
        att2 = attention2 * db2
        att2 = F.avg_pool2d(att2, kernel_size=28, stride=1).view(x.size(0), -1)
        att2 = self.classifier2(att2)

        db3 = self.denseblock3(self.transition2(db2))
        attention3 = self.sigmoid(self.everyconv2dblock1024_1(db3))
        # print_attention(x, attention3, gt, current_timestamp, custom_label='1024a')
        att3 = attention3 * db3
        att3 = F.avg_pool2d(att3, kernel_size=14, stride=1).view(x.size(0), -1)
        att3 = self.classifier3(att3)

        db4 = self.denseblock4(self.transition3(db3))
        attention4 = self.sigmoid(self.everyconv2dblock1024_2(db4))
        # print_attention(x, attention4, gt, current_timestamp, custom_label='1024b')
        att4 = attention4 * db4

        db4 = F.relu(db4, inplace=True)
        db4 = F.avg_pool2d(db4, kernel_size=7, stride=1).view(x.size(0), -1)

        db4 = self.classifier(db4)
 
        return db4, att1, att2, att3

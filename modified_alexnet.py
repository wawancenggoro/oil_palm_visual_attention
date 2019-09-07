import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# VA Alenxet
# =================================================================
class AlexNetVa(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetVa, self).__init__()

        self.conv2d_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2d_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2d_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv2d_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # vanilla linear visual attention
        self.valinear = nn.Linear(256 * 6 * 6, 36)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # manual alexnet
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        # ===========================================
        
        # vanilla visual attention
        va = self.valinear(features.view(-1, 256 * 6 * 6))
        features = features * va.view(features.size()[0], 1, features.size()[2], features.size()[3])

        # ================================================

        features = self.avgpool(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.classifier(features)
        return features

# REVA Alenxet
# =================================================================
class AlexNetReva(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetReva, self).__init__()

        self.conv2d_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2d_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2d_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv2d_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # revisualattention
        # 6 to 13 to 27
        self.transconv2d_1 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=0)
        self.transconv2d_2 = nn.ConvTranspose2d(256, 384, 3, stride=1, padding=1)
        self.transconv2d_3 = nn.ConvTranspose2d(384, 192, 3, stride=1, padding=1)
        self.transconv2d_4 = nn.ConvTranspose2d(192, 64, 3, stride=2, padding=0)
        # 27 to 56 to 112 to 224
        self.transconv2d_5 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0)
        self.transconv2d_6 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.transconv2d_7 = nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # manual alexnet
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        # ===========================================

        # upsampling visual attention
        # 6 to 13 to 27
        va = self.transconv2d_1(features)
        va = self.transconv2d_2(va)
        va = self.transconv2d_3(va)
        va = self.transconv2d_4(va)
        # 27 to 56 to 112 to 224
        va = self.transconv2d_5(va)
        va = self.transconv2d_6(va)
        va = self.transconv2d_7(va)
        x = x * va

        # repeat manual alexnet here
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        # ================================================

        features = self.avgpool(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.classifier(features)
        return features

# FP Alexnet
# =================================================================
class AlexNetFP(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetFP, self).__init__()

        self.conv2d_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2d_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2d_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv2d_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # feature pyramid transconv2d
        # 6 to 13 to 27
        self.transconv2d_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1)
        self.transconv2d_2 = nn.ConvTranspose2d(256, 384, 3, stride=1, padding=1)
        self.transconv2d_3 = nn.ConvTranspose2d(384, 192, 3, stride=2, padding=0)
        # 27 to 55 to 112 to 224
        self.transconv2d_4 = nn.ConvTranspose2d(192, 64, 3, stride=2, padding=0)
        self.transconv2d_5 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=0)
        self.transconv2d_6 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

        # feature pyramid conv2d1x1
        self.conv2d1x1_f4 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv2d1x1_f3 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv2d1x1_f2 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv2d1x1_f1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # manual alexnet
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        # ===========================================

        # upsampling feature pyramid
        f1 = self.relu(self.conv2d_0(x))
        f2 = self.relu(self.conv2d_3(self.maxpool2d(f1)))
        f3 = self.relu(self.conv2d_6(self.maxpool2d(f2)))
        f4 = self.relu(self.conv2d_8(f3))
        f5 = self.relu(self.conv2d_10(f4))

        fp4 = self.transconv2d_1(f5) + self.conv2d1x1_f4(f4)
        fp3 = self.transconv2d_2(f4) + self.conv2d1x1_f3(f3)
        fp2 = self.transconv2d_3(f3) + self.conv2d1x1_f2(f2)
        fp1 = self.transconv2d_4(f2) + self.conv2d1x1_f1(f1)
        fp1 = self.transconv2d_6(self.transconv2d_5(fp1))
        x = x * fp1

        # repeat manual alexnet here
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        # ================================================

        features = self.avgpool(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.classifier(features)
        return features

# StartVA Alexnet
# =================================================================
class AlexNetStart(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetStart, self).__init__()
        self.conv2d_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2d_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2d_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv2d_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.startconv2d = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        attention = self.startconv2d(x)
        x = attention * x

        # manual alexnet
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)

        features = self.avgpool(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.classifier(features)
        return features

# Every Alexnet
# =================================================================
class AlexNetEvery(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetEvery, self).__init__()
        self.conv2d_0 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2d_3 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2d_6 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv2d_8 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv2d_10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.everyconv2dblock64 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock192 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.everyconv2dblock256 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # manual alexnet
        features = self.conv2d_0(x)
        features = self.relu(features)
        features = self.maxpool2d(features)
        attention = F.relu(self.everyconv2dblock64(features))
        features = features + attention

        features = self.conv2d_3(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        attention = F.relu(self.everyconv2dblock192(features))
        features = features + attention

        features = self.conv2d_6(features)
        features = self.relu(features)

        features = self.conv2d_8(features)
        features = self.relu(features)

        features = self.conv2d_10(features)
        features = self.relu(features)
        features = self.maxpool2d(features)
        attention = F.relu(self.everyconv2dblock256(features))
        features = features + attention

        features = self.avgpool(features)
        features = features.view(features.size(0), 256 * 6 * 6)
        features = self.classifier(features)
        return features


# ==============================================================
# ==============================================================

def alexnet(type, pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if (type == "va-alexnet"):
        model = AlexNetVa(**kwargs)
    elif (type == "reva-alexnet"):
        model = AlexNetReva(**kwargs)
    elif (type == "fp-alexnet"):
        model = AlexNetFP(**kwargs)
    elif (type == "start-alexnet"):
        model = AlexNetStart(**kwargs)
    elif (type == "every-alexnet"):
        model = AlexNetEvery(**kwargs)

    if pretrained:
        conv2d_pattern = re.compile(r'(features\.)((0|3|6|8|10)\.(weight|bias))')
        classifier_pattern = re.compile(r'classifier\.(1|4|6)\.(weight|bias)')

        origin_model = model_zoo.load_url(model_urls['alexnet'])

        for key in list(origin_model.keys()):
            conv2d_res = conv2d_pattern.match(key)
            classifier_res = classifier_pattern.match(key)

            if conv2d_res:
                origin_model['conv2d_' + conv2d_res.group(2)] = origin_model[key]
            elif classifier_res:
                origin_model[classifier_res.group()] = origin_model[key]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        origin_model = {k: v for k, v in origin_model.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(origin_model)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


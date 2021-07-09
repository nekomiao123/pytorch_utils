"""
Implementation of our model
"""

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# efficientnet
from efficientnet_pytorch import model as enet

# efficientnet
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=3, bias=False)

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.extract(x)
        x = self.myfc(x)
        return x

# dual pooling
class res18(nn.Module):
    def __init__(self, num_classes):
        super(res18, self).__init__()
        self.base = models.resnet34(pretrained=True)
        self.feature = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduce_layer = nn.Conv2d(1024, 512, 1)
        self.fc  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        bs = x.shape[0]
        x = self.feature(x)
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.reduce_layer(x).view(bs, -1)
        logits = self.fc(x)
        return logits


# frozen the layers or not 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(num_classes, model_name, feature_extract = False, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
        input_size = 224

    elif model_name == 'dualres':
        """ Dual pooling Resnet18
        """
        model_ft = res18(num_classes)

    elif model_name == 'effnet':
        """ efficientnet
        """
        model_ft = enetv2(backbone='efficientnet-b2',out_dim=num_classes)

    elif model_name == 'effnetv2':
        """ efficientnetv2
        """
        print("using efficientnetv2")
        model_ft = timm.create_model('tf_efficientnet_b5_ns', pretrained=True, num_classes=num_classes)

    elif model_name == 'seresnext':
        """ seresnext-50
        """
        print("using se-resnext-50")
        model_ft = timm.create_model('seresnext50_32x4d', pretrained=True, num_classes=num_classes)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs,num_classes))
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs,num_classes))
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


if __name__ == '__main__':
    model = initialize_model(176, 'effnetv2')
    print(model)


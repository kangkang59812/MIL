import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        vgg = models.vgg16(pretrained=True)
        modules = list(vgg.features[i] for i in range(29))
        self.vgg = nn.Sequential(*modules)

    def forward(self, images):
        with torch.no_grad():
            features = self.vgg(images)
        # N,C,H,W=features.size()
        # features=features.view(N,C,H*W)
        # features=features.permute(0,2,1)
        return features


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=512, out_channels=4096, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=4096, out_channels=4096, kernel_size=3)
        self.relu0 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=4096, out_channels=1024, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.pool_mil = nn.MaxPool2d(8, stride=0)

    def forward(self, features):
        # (-1,512,14,14)
        N, C, H, W = features.size()
        relu0 = self.relu0(features)
        conv1 = self.conv1(relu0)
        relu1 = self.relu1(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu2(conv2)
        conv3 = self.conv3(relu2)

        # (-1,1024,8,8)
        sigmoid = self.sigmoid(conv3)
        # (-1,1024,1,1)
        pool = self.pool_mil(sigmoid)
        # (-1,1024)
        x = pool.squeeze(2).squeeze(2)
        # (-1,1024,64)
        x1 = torch.add(torch.mul(sigmoid.view(x.size(0), 1024, -1), -1), 1)
        # (-1,1024)
        cumprod = torch.prod(x1, 2)
        # (-1,1024), why calculate min? 保持子集精度？
        out = torch.min(x, torch.add(torch.mul(cumprod, -1), 1))
        # print('out', out[0])
        return out


import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision



### 残差模块---ResNet 中一个跨层直连的单元
class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super().__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right=nn.Identity()
        # 如果输入和输出的通道不一致，或其步长不为 1，需要将二者转成一致
        if stride!=1 or inchannel!=outchannel:
            self.right=nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self,x):
        left=self.left(x)
        right=self.right(x)
        out=left+right
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self,block,num_classes=10):
        super().__init__()
        self.in_channel=64
        self.conv1=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layers1=self.make_layer(block,channels=64,num_blocks=2,stride=1)
        self.layers2=self.make_layer(block,channels=128,num_blocks=2,stride=2)
        self.layers3=self.make_layer(block,channels=256,num_blocks=2,stride=2)
        self.layers4=self.make_layer(block,channels=512,num_blocks=2,stride=2)
        self.adaptive_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)

    def make_layer(self,block,channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for s in strides:
            layers.append(block(self.in_channel,channels,s))
            self.in_channel=channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.conv1(x)
        out=self.layers1(out)
        out=self.layers2(out)
        out=self.layers3(out)
        out=self.layers4(out)
        out=self.adaptive_pool(out)
        out=out.view(out.shape[0],-1)
        out=self.fc(out)
        return out

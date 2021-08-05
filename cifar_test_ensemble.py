import torch
import torchvision
from torchvision.transforms import transforms
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from model.cifar import *
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from shutil import copyfile
import os
from torch.optim.lr_scheduler import *
from torch.nn import DataParallel
import copy





class ModelEmsemble(nn.Module):
    def __init__(self,model,weights_path):
        super().__init__()
        self.n=len(weights_path)
        self.models=nn.ModuleList(copy.deepcopy(model) for i in range(len(weights_path)))
        for i in range(len(weights_path)):
            data = torch.load(weights_path[i])
            self.models[i].load_state_dict(data['state_dict'], strict=False)
    
    def forward(self,dataloader):
        total=0.0
        correct=0.0

        for i,(img,label) in enumerate(tqdm(dataloader)):
            img,label=img.to(device),label.to(device)
            out_ensemble = []
            for i in range(self.n):
                out_ensemble.append(self.models[i](img))
                out=torch.mean(torch.stack(out_ensemble,dim=0),0)
            prob,pred=torch.max(out,1)
            correct+=(pred==label).sum()
            total+=label.size(0)
        acc=100.0*correct/total
        return acc



if __name__ == '__main__':
    device = torch.device('cuda')

    parser=argparse.ArgumentParser('Pytorch-Image-Classification')
    parser.add_argument('--dataset',type=str,choices=['CIFAR10','CIFAR100'],default='CIFAR10')
    parser.add_argument('--net',type=str,choices=['LeNet','ResNet'],default='ResNet')
    parser.add_argument('--datapath',type=str,default='./data')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--num_workers',type=int,default=0)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--gpu', type=str, default='0,1,2,3' ,help="gpu choose, eg. '0,1,2,...' ")
    args=parser.parse_args()

    #定义对图像预处理的操作
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    #定义训练集和测试集
    if(args.dataset=='CIFAR10'):
        trainset=torchvision.datasets.CIFAR10(args.datapath,train=True,download=False,transform=transform_train)
        testset=torchvision.datasets.CIFAR10(args.datapath,train=False,download=False,transform=transform_test)
    elif(args.dataset=='CIFAR100'):
        trainset=torchvision.datasets.CIFAR100(args.datapath,train=True,download=False,transform=transform_train)
        testset=torchvision.datasets.CIFAR100(args.datapath,train=False,download=False,transform=transform_test)

    #定义Dataloader
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)

    #定义类别信息
    if(args.dataset=='CIFAR10'):
        classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    elif(args.dataset=='CIFAR100'):
        classes={19: 'cattle', 29: 'dinosaur', 0: 'apple', 11: 'boy', 1: 'aquarium_fish', 86: 'telephone', 90: 'train', 28: 'cup', 23: 'cloud', 31: 'elephant', 39: 'keyboard', 96: 'willow_tree', 82: 'sunflower', 17: 'castle', 71: 'sea', 8: 'bicycle', 97: 'wolf', 80: 'squirrel', 74: 'shrew', 59: 'pine_tree', 70: 'rose', 87: 'television', 84: 'table', 64: 'possum', 52: 'oak_tree', 42: 'leopard', 47: 'maple_tree', 65: 'rabbit', 21: 'chimpanzee', 22: 'clock', 81: 'streetcar', 24: 'cockroach', 78: 'snake', 45: 'lobster', 49: 'mountain', 56: 'palm_tree', 76: 'skyscraper', 89: 'tractor', 73: 'shark', 14: 'butterfly', 9: 'bottle', 6: 'bee', 20: 'chair', 98: 'woman', 36: 'hamster', 55: 'otter', 72: 'seal', 43: 'lion', 51: 'mushroom', 35: 'girl', 83: 'sweet_pepper', 33: 'forest', 27: 'crocodile', 53: 'orange', 92: 'tulip', 50: 'mouse', 15: 'camel', 18: 'caterpillar', 46: 'man', 75: 'skunk', 38: 'kangaroo', 66: 'raccoon', 77: 'snail', 69: 'rocket', 95: 'whale', 99: 'worm', 93: 'turtle', 4: 'beaver', 61: 'plate', 94: 'wardrobe', 68: 'road', 34: 'fox', 32: 'flatfish', 88: 'tiger', 67: 'ray', 30: 'dolphin', 62: 'poppy', 63: 'porcupine', 40: 'lamp', 26: 'crab', 48: 'motorcycle', 79: 'spider', 85: 'tank', 54: 'orchid', 44: 'lizard', 7: 'beetle', 12: 'bridge', 2: 'baby', 41: 'lawn_mower', 37: 'house', 13: 'bus', 25: 'couch', 10: 'bowl', 57: 'pear', 5: 'bed', 60: 'plain', 91: 'trout', 3: 'bear', 58: 'pickup_truck', 16: 'can'}

    #定义网络
    if(args.net=='LeNet'):
        net=LeNet(class_num=len(classes)).to(device)
        net.eval()
    elif(args.net=="ResNet"):
        net=ResNet(ResidualBlock,num_classes=len(classes)).to(device)
        net.eval()
    gpus = [ int(_) for _ in args.gpu.split(',') ]
    if(len(gpus)>1):
        net=DataParallel(net,device_ids=gpus, output_device=gpus[0])



    best_acc=0.0
    start_epoch = 0

    weights_path=[
        'saved_models/image_clasification_cifar10_ResNet_1234_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_327_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_1989_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_2000_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_6666_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_9999_best.pth',
        'saved_models/image_clasification_cifar10_ResNet_12321_best.pth',

    ]

    ensemble_model=ModelEmsemble(model=net,weights_path=weights_path)

    acc=ensemble_model(testloader)
    print('Accuracy is %.2f' % acc.item())




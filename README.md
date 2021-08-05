# Pytorch-Image-Classification

🍀 A Pytorch Codebase for Image Classification.⭐⭐⭐

# CIFAR数据集训练

## 参数解释
- --dataset 用于选择数据集，choices=['CIFAR10','CIFAR100']
- --datapath 数据集的路径
- --exp_name 实验的名称，用于保存模型和tensorboard时候的名称
- --logs_folder tensorboard文件保存的路径
- --batch_size batch的大小
- --momentum 动量
- --epoch 训练的epoch
- --lr 学习率
- --num_workers num_workers的数量
- --seed 随机种子
- --gpu gpu的下标，代码中已经写好了多卡训练
- --resume_last 是否要从最后一个epoch继续训练
- --resume_best 是否要从最好的epoch继续训练

## CIRFAR-10

CIRFAR-10 训练代码

```python
python cifar_train.py --dataset CIFAR10
```

## CIRFAR-100

CIRFAR-100 训练代码

```python
python cifar_train.py --dataset CIFAR100
```

## 网络选择


LeNet 训练代码
```python
python cifar_train.py --net LeNet
```

ResNet18 训练代码
```python
python cifar_train.py --net ResNet
```

---

## **机器学习/深度学习算法/计算机视觉/多模态交流群**

## **群里每天都会进行论文的分享哦！！！**

欢迎大家关注公众号：**FightingCV**

公众号**每天**都会进行**论文、算法和代码的干货分享**哦~

![](./tmpimg/FightingCV.jpg)

已建立**机器学习/深度学习算法/计算机视觉/多模态交流群**微信交流群！

**每天在群里分享一些近期的论文和解析**，欢迎大家一起**学习交流**哈~~~


强烈推荐大家关注[**知乎**](https://www.zhihu.com/people/jason-14-58-38/posts)账号和[**FightingCV公众号**](https://mp.weixin.qq.com/s/sgNw6XFBPcD20Ef3ddfE1w)，可以快速了解到最新优质的干货资源。


![](./tmpimg/wechat.jpg)
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# 这里transform参数不止有totensor这一个，我们需要拎出来,用compose方法把我们想要的transform方法放一起
mytransform=transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128))])
# 比之前多加了一个resize方法，它就是用来整理图片尺寸让所有图片都是128*128
mymask=transforms.Compose([transforms.ToTensor(),transforms.Resize((128,128))])

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

class MaskTransform:
    def __call__(self, mask):
        mask = TF.resize(mask, (128, 128), interpolation=Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1
        return mask

traindata=torchvision.datasets.OxfordIIITPet(
    root='../data',
    split='trainval',
    # 这个相当于MNIST用的train=true
    target_types='segmentation',
    # 体现了分割任务和分类任务的不同！
    # 这个数据集里每张宠物图片有两种标签：
    # 一种是分类标签，就是一个数字，表示"这是第几种品种的猫/狗"。跟你 MNIST 里每张图对应一个数字（0-9）一样。
    # 另一种是分割标签，是一张跟原图同尺寸的 mask 图，每个像素标了 1、2 或 3，分别代表前景（宠物）、背景、边界。
    # 你写 target_types='segmentation'，dataloader 每次返回的就是 (图片, mask图)。不写的话返回的是 (图片, 品种编号)。
    # 你现在要做分割，所以需要 mask 图作为标签，让模型学习"每个像素属于什么"。继续改代码。
    transform=mytransform,
    target_transform=MaskTransform(),
    download=True)

testdata=torchvision.datasets.OxfordIIITPet(
    root='../data',
    split='test',
    target_types='segmentation',
    transform=mytransform,
    target_transform=MaskTransform(),
    download=False)

trainloader = torch.utils.data.DataLoader(traindata, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testdata, batch_size=4, shuffle=False)

class down(nn.Module):
    def __init__(self,inpt,outpt):
        super().__init__()
        self.conv1=nn.Conv2d(inpt,outpt,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(outpt)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(outpt,outpt,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(outpt)
    def forward(self,x):
        x=self.conv1(x)
        x=torch.relu(self.bn1(x))
        x=self.conv2(x)
        x=torch.relu(self.bn2(x))
        skip=x
        x=self.pool(x)
        return x,skip

class bottleneck(nn.Module):
# 中间层（bottleneck）也有 BN，结构跟下采样 block 一样:两次 卷积+BN+ReLU，只是没有池化、没有 skip。
    def __init__(self,inpt,outpt):
        super().__init__()
        self.conv1=nn.Conv2d(inpt,outpt,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(outpt,outpt,kernel_size=3,padding=1)
        self.BN1=nn.BatchNorm2d(outpt)
        self.BN2=nn.BatchNorm2d(outpt)
    def forward(self,x):
        x=torch.relu(self.BN1(self.conv1(x)))
        x=torch.relu(self.BN2(self.conv2(x)))
        return x

class up(nn.Module):
    def __init__(self,inpt,outpt):
        # 不能给init传x,skip这种数据，只有毫无特殊来头的常数死数字可以
        super().__init__()
# 放大是为了  1.和skip拼接，skip是池化前的，池化缩小了一倍。所以想要能cat需要要放大一倍
#           2.为了还原到原图的尺寸
# "那为啥还要池化来着?"  因为每次的池化都是为了后续的卷积能够学到更多,池化缩小图片,相同尺寸卷积核一次学的就更多,这样卷积池化一层一层叠加下去，模型就能从"看局部纹理"变成"看整体结构"。
# sum: 下采样（池化）是为了让模型理解"这是什么"，上采样（放大）是为了还原"在哪里"。两个配合才能做到逐像素的预测。
        self.big=nn.Upsample(scale_factor=2)
        self.conv1=nn.Conv2d(inpt,outpt,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(outpt,outpt,kernel_size=3,padding=1)
        self.BN1=nn.BatchNorm2d(outpt)
        self.BN2=nn.BatchNorm2d(outpt)
    def forward(self,x,skip):
        x=self.big(x)
        x=torch.cat((x,skip),dim=1)
        x=torch.relu(self.BN1(self.conv1(x)))
        x=torch.relu(self.BN2(self.conv2(x)))
        return x

class myunet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1=down(3,64)
        self.down2=down(64,128)
        self.bottleneck=bottleneck(128,256)
        '''
        这里的通道数计算稍微特殊一点点！
        up1: 256 + skip2(128) = 384 → 128
        up2: 128 + skip1(64) = 192 → 64
        '''
        self.up1=up(384,128)
        self.up2=up(192,64)
        '''
        关于unet的任务：
        不是给整张图做分类，而是给每个像素做分类
        所以每次在unet类里要单独定义一个整个前向传播的最后一步:用一个特殊的卷积卷积把通道数压缩到类别数。
        '''
        self.conv=nn.Conv2d(64,3,kernel_size=1)
    #     用 kernel_size=1 就行，这一步只是压通道，不需要 3×3，而且padding要去掉，用不上。

    def forward(self,x):
        x,skip1=self.down1(x)
        x,skip2=self.down2(x)
        x=self.bottleneck(x)
        x=self.up1(x,skip2)
        x=self.up2(x,skip1)
        x=self.conv(x)
        return x
        '''
        skip2 传给 up1，skip1 传给 up2。
        因为 up1 是从最底下往上走的第一步，它要拼接的是离它最近的那个 skip，就是最后一个下采样存的 skip2。然后 up2 再拼 skip1。
        下采样是从外往里，上采样是从里往外，skip 的配对是对称的。
        '''

'''
关于层数和通道数:
通道数大小（3→128 vs 3→64）：通道数越大，这一层能学到的特征种类越多。64 个通道就是 64 个不同的卷积核，能检测 64 种不同的 pattern。
    128 就是 128 种。但单层给太多通道有点浪费，因为输入信息就那么多，一层吃不透。
层数多少：这个影响更大。每多一层下采样，感受野翻倍。一层下采样，卷积核只能看很小的局部。两层之后能看到更大范围，才能理解"这一块东西是猫的耳朵"这种稍微全局一点的信息。                                  
    所以层数比通道数重要。一层 3→128 不如两层 3→64→128。
'''
'''
关于尺寸:
1.池化和放大(upsample)都不影响通道数，只影响图像尺寸
2.卷积会改变通道数，通道数代表学到的特征种类。
注意卷积不一定放大通道数，具体通道数想怎么变完全是人为决定的。比如上采样层的通道数就是变小。原因:
(！！！超绝冷知识！！！)通道数和尺寸有点点动态平衡的意思: 下采样时通道越来越大是因为尺寸在缩小，用更多通道来补偿信息。上采样时尺寸在恢复，不需要那么多通道了，所以压回去。对称的。
可能是为了防止信息丢失之类的，尺寸对应的是语义信息、通道对应的是空间信息，一个少一点另一个就补一点。反正这个东西Claude说不用想太多
'''

MYunet=myunet()
# 不要忘了实例化，训练的时候我们用实例来传递处理数据而不是类
cross=nn.CrossEntropyLoss()
sgd=torch.optim.SGD(MYunet.parameters(),lr=0.01)

img, mask = traindata[0]
print(mask.unique())

epochs=10
for epoch in range(epochs):
    for pics,mask in trainloader:
        sgd.zero_grad()
        out=MYunet(pics)
        loss=cross(out,mask)
        loss.backward()
        sgd.step()

    with torch.no_grad():
        total = 0
        correct = 0
        for pics,mask in testloader:
            out=MYunet(pics)
# 其实没什么特别的，之前的分类任务是统计ans正确率，因为我们一开始下载加载的标签就是ans，这里我们之前下载加载的是mask，所以只是把ans换成mask，改个名字而已。
            total += mask.numel()
            # 不是total+=mask.size(0)。mask.numel() 是 mask 里所有像素的数量，不是 mask.size(0)（那个是 batch 数）。
            predict = torch.argmax(out, dim=1)
            # 漏写predict，然后下面correct里的predict==mask写成out==mask
            correct+=(predict==mask).sum().item()
        accuracy=correct/total*100
        print(f"正确率:{accuracy}%")

'''
with torch.no_grad():
    correct = 0
    total = 0
    for pics, mask in testloader:
        out = model(pics)
        predict = torch.argmax(out, dim=1)  # (batch, H, W)
        correct += (predict == mask).sum().item()
        total += mask.numel()
    accuracy = correct / total * 100
'''




















import numpy as np
print(np.__version__)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
trainset=torchvision.datasets.MNIST(
    root='./data',
    transform=transforms.ToTensor(),
    download=True,
    train=True
)
testset=torchvision.datasets.MNIST(
    root='./data',
    transform=transforms.ToTensor(),
    train=False
)

trainload=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True)
# 每次都忘了加batchsize!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
testload=torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        # 写init的时候不要忘了把大家都挂在self上

        super().__init__()
        self.juanji1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        # 步长默认是1
        self.juanji2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        # 一张图片被32个滤波器各处理了一遍，每个滤波器对应一个通道，每个通道对应一张输出图
        # 滤波器有几个核是自动决定的，不需要设置。通道数和输入图片数量无关
        #
        # 没有设置输入图像尺寸的地方，Conv2d不管图片大小。
        # 它只关心通道数和卷积核大小，输入图片是28×28还是32×32都能处理，输出尺寸由输入尺寸自动决定。也不管图像被分成几份，自动决定。
        # 所以同一个模型可以处理不同尺寸的图片，尺寸是数据决定的，不是层定义的。

        #  1  →  32  →  64
        # 灰度  第一层  第二层

        # 图片的颜色用数字表示：
        #
        # 灰度图：每个像素只有1个数字（0=黑，255=白），所以1个通道
        # 彩色图：每个像素有3个数字（R、G、B各一个），所以3个通道

        self.chihua=nn.MaxPool2d(kernel_size=2)
        # MaxPool2d默认stride=kernel_size，所以kernel_size=2时步长自动就是2，不用手动写。
        # Conv2d默认stride=1，步长1配合padding=1就能保持尺寸不变，也不用写。
        self.layer1=nn.Linear(3136,128)
        self.layer2=nn.Linear(128,10)
    def forward(self,x):
        x=torch.relu(self.juanji1(x))
        x=self.chihua(x)
        x=torch.relu(self.juanji2(x))
        x = self.chihua(x)
        x=x.view(-1,3136)
        x=torch.relu(self.layer1(x))
        x=self.layer2(x)
        return x

mycnn=MLP()
cross=nn.CrossEntropyLoss()
# 内置了softmax
sgdoptim=torch.optim.SGD(mycnn.parameters(),lr=0.01)

epochs=10

for epoch in range(epochs):
    alloss=0
    for pics,ans in trainload:
        sgdoptim.zero_grad()
        fwdrst=mycnn(pics)
        loss=cross(fwdrst,ans)
        loss.backward()
        sgdoptim.step()
        alloss+=loss.item()

    crct=0
    tt=0
    with torch.no_grad():
        for pics,ans in testload:
            fwdrst=mycnn(pics)
            guess=torch.argmax(fwdrst,dim=1)
            crct+=(guess==ans).sum().item()
            tt+=ans.size(0)
    accuracy=crct/tt
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {alloss / len(trainload):.4f} | 测试准确率: {accuracy * 100:.2f}%")

# BN怎么运作：
# 每个batch里，对每个通道的所有数值算均值和方差，然后把数值拉回到均值0、方差1附近：
# 原来：[0.1, 500, -300, 0.02, ...]  # 数值乱七八糟
# BN后：[-0.1, 1.2, -0.8, 0.05, ...]  # 稳定在0附近
# 防止某些层的数值爆炸或消失，训练更稳定。

# 两个ResBlock叠加怎么运作：
# 输入x1
# ↓
# ResBlock1 → 输出x2（学到了边缘、纹理等简单特征）
# ↓
# ResBlock2 → 输出x3（在x2基础上学到更复杂的组合特征）
# 每个ResBlock内部还是"卷积结果+输入"，只是第二个ResBlock的输入是第一个的输出。

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train=torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test=torchvision.datasets.MNIST(
    root='../data',
    train=False,
    transform=transforms.ToTensor(),
)
trainloader = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)
testloader = torch.utils.data.DataLoader(test,batch_size=32,shuffle=False)

class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(channels)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
    def forward(self,x):
        tempt=x
# PyTorch规定必须叫forward： 因为你调用model(x)的时候，PyTorch内部自动去找forward这个方法执行。叫别的名字它找不到。
        x=self.bn1(self.conv1(x))
        x=torch.relu(x)
        x=self.bn2(self.conv2(x))
        x=torch.relu(x)+tempt
# 注意要加“原来的x”，要加tempt！！！
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,stride=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,stride=1)
        self.bn1=nn.BatchNorm2d(32)
        self.bn2=nn.BatchNorm2d(64)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.blk1 = ResBlock(32)
        self.blk2 = ResBlock(32)
        self.blk3 = ResBlock(64)
        self.blk4 = ResBlock(64)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(64,128)
        self.fc2=nn.Linear(128,10)
    def forward(self,x):
        x=self.conv1(x)
        x=torch.relu(self.bn1(x))
        x=self.pool(x)
        x=self.blk1(x)
        x=self.blk2(x)
        x=self.conv2(x)
        x=torch.relu(self.bn2(x))
        x=self.pool(x)
        x=self.blk3(x)
        x=self.blk4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x

MYresnet=ResNet()

cross=nn.CrossEntropyLoss()
sgd=torch.optim.SGD(MYresnet.parameters(),lr=0.01)

epochs=10
for epoch in range(epochs):
    for pics,ans in trainloader:
        sgd.zero_grad()
        # 勿忘每批训练的第一步是清空参数
        outcome=MYresnet(pics)
        loss=cross(outcome,ans)
        loss.backward()
        sgd.step()

    correct=0
    total=0

    with torch.no_grad():
        for pics,ans in testloader:
            outcome=MYresnet(pics)
            predict=torch.argmax(outcome,dim=1)
            correct+=(predict==ans).sum().item()
            # 勿忘累加
            total+=ans.size(0)
            # 总数是答案的size0的累加。标签的唯一形状就是标签的数量，但是输出不是它是二维数组，所以不能写outcome.size(0)
        accuracy=(correct/total)*100
        print(f"正确率{accuracy}%")
        # 注意accuracy和print是一批一批的，要写在循环外边

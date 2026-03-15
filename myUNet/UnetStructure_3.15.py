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
trainloader=torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)
testloader=torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)

class downandmiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.BN1=nn.BatchNorm2d(128)
        self.BN2=nn.BatchNorm2d(64)
        self.BN3=nn.BatchNorm2d(32)
        self.BN4=nn.BatchNorm2d(64)
    def forward(self, x):
        x=self.conv1(x)
        x = torch.relu(self.BN3(x))
        skip1=x
        x=self.pool(x)

        x=self.conv2(x)
        x = torch.relu(self.BN4(x))
        skip2=x
        x = self.pool(x)

        # 中间层
        x=self.conv3(x)
        x = torch.relu(self.BN1(x))
        x=self.conv4(x)
        x=torch.relu(self.BN2(x))
        return x,skip1,skip2
#  最后一次输出是64通道，通道里每张特征图的尺寸是7*7。
#  然后上采样变成14×14（上采样就是扩大，比如最近邻插值）还是64通道，拼接skip2（也是64通道），合并后变成128通道。


class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        # 比较简单的最近邻插值，找最近的像素复制
        # 创建一个上采样层，scale_factor=2就是把尺寸放大2倍，7→14，14→28。
        # 和nn.MaxPool2d(kernel_size=2)是反操作，一个缩小一个放大。

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(32)
        self.outconv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x,skip1,skip2):
        x=self.up(x)
        x=torch.cat([x,skip2],dim=1)
# 通过cat拼接一些细节上去(skip2):
#   拼接skip2把编码器在14×14时保存的原始特征图贴进来，通道从64变128，相当于给解码器同时提供了两份信息：
#       x：全局的、抽象的特征（从中间层来的）
#       skip2：局部的、细节的特征（编码器原始保存的）
#   后面的卷积综合这两份信息，输出既有全局又有细节的特征图。
        x=torch.relu(self.bn1(self.conv1(x)))
        # 拼接之后要卷积，目的是让模型学习如何融合x和skip的信息，不是简单拼在一起就完了。
        # 卷积把128通道压回64，同时提取融合后的特征。融合的作用

        x=self.up(x)
        x=torch.cat([x,skip1],dim=1)
        x=torch.relu(self.bn2(self.conv2(x)))

        x=self.outconv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=downandmiddle()
        self.decoder=decoder()
    def forward(self,x):
        x,skip1,skip2=self.encoder(x)
        x=self.decoder(x,skip1,skip2)
        return x

model = UNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for pics, _ in trainloader:
        optimizer.zero_grad()
        output = model(pics)
        loss = criterion(output, pics)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(trainloader):.4f}")

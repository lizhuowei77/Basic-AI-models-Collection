import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math


T=300
beta1=1e-4
beta2=0.02
mybeta=torch.linspace(beta1,beta2,T)

myalpha=1.0-mybeta

alphac=torch.cumprod(myalpha,dim=0)

mytransform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

trainset=torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=mytransform,
    download=True
)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True,)

# ----------------------------------------------------------------------------------------------------------------------

def addnoise(x0,t):
    noise=torch.randn_like(x0)
    sgpica=alphac[t]
    sgpica=sgpica.view(-1,1,1,1)
    xt = torch.sqrt(sgpica) * x0 + torch.sqrt(1.0 - sgpica) * noise
    return xt,noise

# ----------------------------------------------------------------------------------------------------------------------

class embed(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.dim=dim
    def forward(self,t):
        half=self.dim//2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args=t.float().unsqueeze(1)*freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return embedding

# ----------------------------------------------------------------------------------------------------------------------

class down(nn.Module):
    def __init__(self,dim,in_,out_):
        super().__init__()
        self.conv1=nn.Conv2d(in_,out_,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(out_,out_,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_)
        self.bn2=nn.BatchNorm2d(out_)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.proj=nn.Linear(dim,out_)
    def forward(self,x,embedt):
        x=torch.relu(self.bn1(self.conv1(x)))
        t=self.proj(embedt).unsqueeze(2).unsqueeze(3)
        x=x+t
        x=torch.relu(self.bn2(self.conv2(x)))
        skip=x
        x=self.pool(x)
        return x,skip

class mid(nn.Module):
    def __init__(self,dim,in_,out_):
        super().__init__()
        self.conv1=nn.Conv2d(in_,out_,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(out_,out_,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_)
        self.bn2=nn.BatchNorm2d(out_)
        self.proj=nn.Linear(dim,out_)
    def forward(self,x,embedt):
        x=torch.relu(self.bn1(self.conv1(x)))
        t=self.proj(embedt).unsqueeze(2).unsqueeze(3)
        x=x+t
        x=torch.relu(self.bn2(self.conv2(x)))
        return x

class up(nn.Module):
    def __init__(self,dim,in_,out_):
        super().__init__()
        self.upsp = nn.Upsample(scale_factor=2)
        self.conv1=nn.Conv2d(in_,out_,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(out_,out_,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_)
        self.bn2=nn.BatchNorm2d(out_)
        self.proj=nn.Linear(dim,out_)
    def forward(self,x,embedt,skip):
        x=self.upsp(x)
        x=torch.cat((x,skip),dim=1)
        x=torch.relu(self.bn1(self.conv1(x)))
        t=self.proj(embedt).unsqueeze(2).unsqueeze(3)
        x=x+t
        x=torch.relu(self.bn2(self.conv2(x)))
        return x

# ----------------------------------------------------------------------------------------------------------------------

class DiffusionUnet(nn.Module):
    def __init__(self):
        super().__init__()
        dim=64
        self.embedt=embed(dim)
        self.down1=down(dim,1,32)
        self.down2=down(dim,32,64)
        self.midle=mid(dim,64,128)
        self.up1=up(dim,192,64)
        self.up2=up(dim,96,32)
        self.squeeze=nn.Conv2d(32,1,kernel_size=1)

    def forward(self,x,t):
        et=self.embedt(t)
        x,skip1=self.down1(x,et)
        x,skip2=self.down2(x,et)
        x=self.midle(x,et)
        x=self.up1(x,et,skip2)
        x=self.up2(x,et,skip1)
        x=self.squeeze(x)
        return x

myDDPMunet=DiffusionUnet()

# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def sample(myDDPMunet,picnum=16):
    myDDPMunet.eval()
    x=torch.randn(picnum,1,28,28)

    for step in reversed(range(T)):
        picstep=torch.full((picnum,),step,dtype=torch.long)
        predict=myDDPMunet(x,picstep)

        beta_t = mybeta[step]
        alpha_t = myalpha[step]
        alphaccum_t = alphac[step]

        coef = beta_t / torch.sqrt(1.0 - alphaccum_t)
        x = (1.0 / torch.sqrt(alpha_t)) * (x - coef * predict)

        if step > 0:
            sigma = torch.sqrt(beta_t)  # sigma 不是噪声本身，是控制加多少噪声的系数(很tiny)。
            x = x + sigma * torch.randn_like(x)

    myDDPMunet.train()
    return x

# ----------------------------------------------------------------------------------------------------------------------

optimizer = torch.optim.Adam(myDDPMunet.parameters(), lr=1e-3)
mse = nn.MSELoss()

epochs = 5
'''
实际 DDPM 训练逻辑是:
对单批次的每张图片，直接随机选一个步数 t（比如有的选 50 步、有的选 800 步），
直接模拟 “加 t 步噪声后的结果”，让模型直接学习 “从 t 步噪声图还原”，而非逐步递进。
'''
for epoch in range(epochs):
    total_loss = 0
    count = 0

    for pics, labels in trainloader:
        optimizer.zero_grad()
        t = torch.randint(0, T, (pics.size(0),))

# 给每张图片随机选一个步数，直接模拟加噪到那一步时的情形，直接练习预测所选这一步（只有这一步）的噪声，然后预测的和这一步对应的真实加上去的噪声比对。


        # 2. 加噪
        xt, noise = addnoise(pics, t)
        # xt: 加噪后的图, noise: 真实噪声（待会算loss要用）

        # 3. 模型预测噪声
        pred_noise = myDDPMunet(xt, t)

        # 4. 算loss：预测噪声 vs 真实噪声
        loss = mse(pred_noise, noise)

        # 5. 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count = count+1

    avg_loss = total_loss / count
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

print("开始生成图片...")
generated = sample(myDDPMunet, picnum=16)
# generated: (16, 1, 28, 28)，值可能超出[-1,1]，需要clamp

# 从[-1,1]转回[0,1]用于保存
generated = (generated.clamp(-1, 1) + 1) / 2

# 用torchvision的工具把16张图拼成一个4x4的网格图保存
from torchvision.utils import save_image

save_image(generated, 'generated_mnist2.png', nrow=4)
print("生成完成！保存为 generated_mnist2.png")

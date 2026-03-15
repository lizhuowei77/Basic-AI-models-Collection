import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

traindata=torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True,
)
testdata=torchvision.datasets.MNIST(
    root='../data',
    train=False,
    transform=transforms.ToTensor()
)
loadtrain=torch.utils.data.DataLoader(traindata,batch_size=32,shuffle=True)
loadtest=torch.utils.data.DataLoader(testdata,batch_size=32,shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2)
        # 池化的参数是kernel_size=2,有点记不住
        # 入通道数————滤波器数量————上一个的输出图通道数

        self.lay1=nn.Linear(3136,128)
        self.lay2=nn.Linear(128,10)
    def forward(self,x):
        x=self.conv1(x)
        x=torch.relu(self.pool(x))
        x=self.conv2(x)
        x=torch.relu(self.pool(x))
        x=x.view(-1,3136)
        # 3136=7*7的图片再乘通道数64
        x=torch.relu(self.lay1(x))
        x=self.lay2(x)
        return x

mycnn=MLP()
crocs=nn.CrossEntropyLoss()
myoptim=torch.optim.SGD(mycnn.parameters(),lr=0.01)

epochs=10
for epoch in range(epochs):
    totalloss=0
    for pics,ans in loadtrain:
        myoptim.zero_grad()
        rst=mycnn(pics)
        loss=crocs(rst,ans)
        loss.backward()
        myoptim.step()

    right=0
    total=0
    with torch.no_grad():
# 有点记不住torch.no_grad()
        for pics,ans in loadtest:
            outcome=mycnn(pics)
            predct=torch.argmax(outcome,dim=1)
            right+=(predct==ans).sum().item()
            total+=ans.size(0)
        accuracy=right/total
        print(f"准确率{accuracy}")


























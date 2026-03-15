import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

trainset=torchvision.datasets.MNIST(
    root='../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
testset=torchvision.datasets.MNIST(
    root='../data',
    train=False,
    transform=transforms.ToTensor()
)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=32,shuffle=True)
# 不要忘了设置batchsize
testloader=torch.utils.data.DataLoader(testset,batch_size=32,shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1=nn.Linear(784,124)
        self.lay2=nn.Linear(124,10)
    def forward(self,x):
        x=x.view(-1,784)
        x=torch.relu(self.lay1(x))
        x=self.lay2(x)
        return x

mymlp=MLP()
cross=nn.CrossEntropyLoss()
sgd=torch.optim.SGD(mymlp.parameters(),lr=0.01)

epochs=10
for epoch in range(epochs):
    for pics,ans in trainloader:
        sgd.zero_grad()
        outcm=mymlp(pics)
        loss=cross(outcm,ans)
        loss.backward()
        sgd.step()

    right=0
    all=0
    with torch.no_grad():
        for pics,ans in testloader:
            outcm=mymlp(pics)
            guess=torch.argmax(outcm,dim=1)
            right+=(guess==ans).sum().item()
            all+=ans.size(0)

    accuracy=right/all
    print(f"epoch{epoch}/{epochs}|accuracy{accuracy*100:.2f}%")
















import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math

# ============================================================
# 第一阶段/第一部分:把表准备好
# ============================================================



T = 300  # 总步数，MNIST简单，300步够了（原论文用1000）
beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T)
# torch.linspace 就是生成一组等间距的数。
# linspace(0.0001, 0.02, 300) 意思是：从0.0001到0.02，均匀地生成300个数。第一个是0.0001，最后一个是0.02，中间的数等间距排列。
# 就是给你一把均匀刻度的尺子，从 beta_start 到 beta_end 切成300份。
# betas就是一个长度为T的一维tensor，比如[0.0001, 0.000167, ..., 0.02]

alphas = 1.0 - betas
# alphas[t] = 1 - betas[t]

alpha_bars = torch.cumprod(alphas, dim=0)
# alphas 是300个独立的值：[α₁, α₂, α₃, ..., α₃₀₀]
# alpha_bars 是累积乘积：[α₁, α₁×α₂, α₁×α₂×α₃, ..., α₁×α₂×...×α₃₀₀]



#--------------------------------------------------------------第二阶段:训练---------------------------------------------------------------



# ============================================================
# 第二部分：数据加载
# ============================================================

mytransform = transforms.Compose([
    transforms.ToTensor(),
    # ToTensor已经把像素从[0,255]变成[0,1]了
    # 再归一化到[-1,1]，因为噪声是标准正态分布（均值0），图片也要中心化
    transforms.Normalize((0.5,), (0.5,))
])

traindata = torchvision.datasets.MNIST(
    root='../data', train=True, transform=mytransform, download=True
)
trainloader = torch.utils.data.DataLoader(
    traindata, batch_size=64, shuffle=True
)



# ============================================================
# 第三部分：前向加噪函数
# ============================================================
# 给定干净图片x0和时间步t，一步算出加噪后的xt
# 公式：xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

def add_noise(x0, t, noise=None):
    """
    x0: 干净图片, shape (batch, 1, 28, 28)
    t: 时间步, shape (batch,)，每张图随机一个t
    noise: 可选，如果不传就自动生成随机噪声
    返回：加噪后的图片xt，和用到的噪声epsilon
    """
    if noise is None:
        noise = torch.randn_like(x0)
        # randn_like：生成跟x0形状一样的标准正态随机数

    # 取出每张图对应的alpha_bar值
    ab = alpha_bars[t]  # shape: (batch,)
    # 变形成 (batch, 1, 1, 1) 才能跟 (batch, 1, 28, 28) 的图片做乘法
    ab = ab.view(-1, 1, 1, 1)

    xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
    # (这行只是根号顶)   ____        ______
    # 加噪小公式，xt = √ αˉt ⋅ x0 + √ 1−αˉt · ϵ
    # 按比例混合原图和噪声，t越大噪声越多。
    return xt, noise



# ============================================================
#第四部分：时间步嵌入
# ============================================================
# 模型需要知道"当前是第几步"，因为不同步数的噪声程度不同
# 用sinusoidal embedding把整数t变成一个向量，跟Transformer的位置编码一个原理

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        """dim: 嵌入向量的维度"""
        super().__init__()
        self.dim = dim
# 传进来的dim，如果不写上面这一行的话，只是一个局部变量，出了init就没了。dim不是临时数据所以往init里传

    def forward(self, t):
        """
        t: (batch,) 的整数 tensor
        返回: (batch, dim) 的浮点向量
        """
        half = self.dim // 2
        # 频率从低到高
        freqs = torch.exp(-math.log(10000) * torch.arange (half, dtype=torch.float32) / half)
        # 生成32个频率值。不需要记这个公式，它是从Transformer论文里抄过来的固定写法。你只需要知道结果是一组从大到小的数，代表从低频到高频。
        # t是(batch,)，变成(batch,1)；freqs是(half,)，变成(1,half)
        # 相乘得到 (batch, half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        # 就是每个时间步乘以每个频率，得到一组角度值。然后拿这些角度去算sin和cos。
        # args 这个变量名就是"arguments"的缩写，意思是"传给sin/cos的参数"。
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        # sin和cos拼起来，得到 (batch, dim)
        return embedding

'''
上面这段😭把位置编码一下
'''



# ============================================================
# 第五部分：带时间嵌入的UNet    unet好像就是训练咋识别噪声
# ============================================================
# 跟你之前写的UNet结构一样：下采样 → bottleneck → 上采样 + skip连接
# 区别：每个block多了一步——把时间嵌入加进去

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 下面这行是时间嵌入投影：把time_dim维的向量投影到out_ch维，这样才能跟特征图相加。
        # 这个的意义只是改一下时间的形状，让它可以加到x上:
        self.time_proj = nn.Linear(time_dim, out_ch)
#下:卷积不池化、时间映射、卷积、为（后期解码器）跳跃cat存档skip=x、池化
    def forward(self, x, t_emb):
        """
        x: 图片特征 (batch, in_ch, H, W)
        t_emb: 时间嵌入 (batch, time_dim)
        """
        x = torch.relu(self.bn1(self.conv1(x)))

        # 把时间信息加进来
        # time_proj(t_emb) 的shape是 (batch, out_ch)
        # 变成 (batch, out_ch, 1, 1) 才能跟 (batch, out_ch, H, W) 相加
        t = self.time_proj(t_emb).unsqueeze(2).unsqueeze(3)
        x = x + t  # 广播相加，每个空间位置都加上同样的时间信息

        x = torch.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        return x, skip


class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
#中:卷积不池化、时间映射、卷积不池化
    def forward(self, x, t_emb):
        x = torch.relu(self.bn1(self.conv1(x)))
        t = self.time_proj(t_emb).unsqueeze(2).unsqueeze(3)
        x = x + t
        x = torch.relu(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
#上:上采样、跳跃cat、卷积不池化、时间映射、卷积不池化
    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        x = torch.relu(self.bn1(self.conv1(x)))
        t = self.time_proj(t_emb).unsqueeze(2).unsqueeze(3)
        x = x + t
        x = torch.relu(self.bn2(self.conv2(x)))
        return x

'''
⭐❤ time_dim 和 time_proj 和 time_embed的关系: ❤🎉
    dim是步数向量的维度，小超参数，决定向量的复杂度(由几个数字组成)。
        它用于传进每一层的init，以及传进unet整合总类里，然后用来得到proj和embed
    proj它不表达任何含义，就是一个维度转换器
        它用于每一层的时间映射。都要用他转换一下
    而embed则是按照dim提供的大小来设置的具体内容嵌入
        它被proj转换维度后，用于每一层的时间映射，加到x上       
'''


class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        time_dim = 64  # 时间嵌入的维度
        # 就是把整数 t 变成多长的向量。
        # t 是一个数，数字小就信息太少。时间嵌入把它展开成64个数的向量，模型能从中提取更丰富的时间信息。

        self.time_embed = TimeEmbedding(time_dim)

        # MNIST是1通道，通道数比你宠物分割那个小一些，CPU跑得动
        self.down1 = DownBlock(1, 32, time_dim)       # 1→32, 28→14
        self.down2 = DownBlock(32, 64, time_dim)       # 32→64, 14→7

        self.bottleneck = BottleNeck(64, 128, time_dim)  # 64→128, 7×7

        # 跟你之前一样的通道数计算：
        # up1: 128 + skip2(64) = 192 → 64
        # up2: 64 + skip1(32) = 96 → 32
        self.up1 = UpBlock(192, 64, time_dim)
        self.up2 = UpBlock(96, 32, time_dim)

        # 最后一层：输出1通道（预测的噪声，跟输入图片通道数一样）
        # 之前宠物分割这里输出3（3个类别），DDPM这里就输出1
        # 输出通道数永远跟输入图片通道数一样，因为你要预测的噪声跟原图形状相同——每个像素加了多少噪声，就预测每个像素的噪声值。
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, t):
        """
        x: 加噪后的图片 (batch, 1, 28, 28)
        t: 时间步 (batch,) 整数
        返回: 预测的噪声 (batch, 1, 28, 28)
        """
        t_emb = self.time_embed(t)  # (batch, time_dim)

        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        x = self.final(x)
        return x

model = DiffusionUNet()

# 冰知识: unet预测的噪声(和t有关的函数)是“图里有什么噪声”而不是具体“要减多少噪声”，具体要减多少的计算就是查表和用公式，我们放到采样函数里。

# ============================================================
# 第六部分：采样函数（推理/生成）
# ============================================================
# 从纯噪声开始，一步步去噪，最终得到生成的图片

@torch.no_grad()  # 推理不需要梯度
def sample(model, n_samples=16):
    """
    model: 训练好的UNet
    n_samples: 要生成几张图
    返回: 生成的图片 (n_samples, 1, 28, 28)
    """
    model.eval()      # nn.module自带的方法，用来切换到评估模式。类似no_grad()

    # 从纯噪声开始
    x = torch.randn(n_samples, 1, 28, 28)

    # 从t=T-1一直去噪到t=0
    for t_val in reversed(range(T)):
        # 这一行在倒着走，加噪是0-299，去噪就是反过来299-0
        t_batch = torch.full((n_samples,), t_val, dtype=torch.long)
        '''
        PyTorch要求shape参数必须是元组。(n_samples,) 是一个元组，n_samples 只是一个整数。
        那个逗号就是Python里表示"这是一个只有一个元素的元组"的写法。(16) 是整数16，(16,) 才是元组。
        '''
        # 创建一个tensor，里面16个数全是同一个值。
        # 比如当前去噪到第200步，t_val=200，这行就生成 [200, 200, 200, ..., 200]（16个200）。
        # 因为你同时生成16张图，每张图需要一个时间步，而它们都在同一步去噪，所以全是同一个数。
        # ⭐就是表示16张图片都在第200步
        # full使用方法:第一个参数是shape，第二个参数是拿什么数填满它。

        # 模型预测噪声
        pred_noise = model(x, t_batch)
        # 把当前的噪声图还有步数喂给unet预测噪声

        # 去噪公式（DDPM论文Algorithm 2）用处在于，照着这个公式我们知道怎么把噪声图上去掉我们预测的噪音
        # 下面这三行是查表取当前步骤的预训练参数
        beta_t = betas[t_val]
        alpha_t = alphas[t_val]
        alpha_bar_t = alpha_bars[t_val]

        # 下面是按比例在当前噪声图里把预测的噪声值减少掉
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * pred_noise) + sigma_t * z
        coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        x = (1.0 / torch.sqrt(alpha_t)) * (x - coef * pred_noise)

        # 每一步去噪完成后加回一点随机噪声（除了最后一步，因为最终要输出干净图片），因为这样能保证生成结果的多样性
        if t_val > 0:
            sigma = torch.sqrt(beta_t)       # sigma 不是噪声本身，是控制加多少噪声的系数(很tiny)。
            x = x + sigma * torch.randn_like(x)

    model.train()
    return x


# ============================================================
# 第七部分：训练
# ============================================================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 用Adam不用SGD，扩散模型训练用Adam更稳定！
mse = nn.MSELoss()
# 是损失函数。之前UNet分割任务等输出的是类别，用CrossEntropy。DDPM输出的是连续数值（噪声），用MSE。

epochs = 5  # MNIST简单，5个epoch就能看到效果

'''
实际 DDPM 训练逻辑是:
对单批次的每张图片，直接随机选一个步数 t（比如有的选 50 步、有的选 800 步），
直接模拟 “加 t 步噪声后的结果”，让模型直接学习 “从 t 步噪声图还原”，而非逐步递进。
'''

for epoch in range(epochs):
    total_loss = 0
    count = 0

    for pics, labels in trainloader:
        # labels在DDPM里完全用不到！我们不关心这张图是数字几
        # 我们只关心：给图加噪 → 让模型预测噪声

        optimizer.zero_grad()

        # 1. 随机给batch里每张图采一个时间步
        t = torch.randint(0, T, (pics.size(0),))
        # torch.randint(0, 300, (64,)) 意思是：生成64个随机整数，每个都在0到299之间。
        # 64是因为 pics.size(0) 就是batch size，一个batch有64张图。每张图随机分配一个时间步，所以需要64个随机数。

        # 2. 加噪
        xt, noise = add_noise(pics, t)
        # xt: 加噪后的图, noise: 真实噪声（待会算loss要用）

        # 3. 模型预测噪声
        pred_noise = model(xt, t)

        # 4. 算loss：预测噪声 vs 真实噪声
        loss = mse(pred_noise, noise)

        # 5. 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# ============================================================
# 第八部分：生成图片并保存
# ============================================================

print("开始生成图片...")
generated = sample(model, n_samples=16)
# generated: (16, 1, 28, 28)，值可能超出[-1,1]，需要clamp

# 从[-1,1]转回[0,1]用于保存
generated = (generated.clamp(-1, 1) + 1) / 2

# 用torchvision的工具把16张图拼成一个4x4的网格图保存
from torchvision.utils import save_image
save_image(generated, 'generated_mnist1.png', nrow=4)
print("生成完成！保存为 generated_mnist1.png")
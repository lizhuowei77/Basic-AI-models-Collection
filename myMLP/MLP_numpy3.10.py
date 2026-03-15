        # 下载 MNIST #

import urllib.request
# url就是网址，就是你在浏览器地址栏里看到的那串https://...
# lib：是 Library（库）的缩写；
# 合起来 urllib 就是「Python 处理网址的工具库」。
# 用来发网络请求、下载文件的，我们用它来下载MNIST数据集
import os
# 用来操作文件系统的，比如创建文件夹、拼接路径，我们用它来创建mnist_data文件夹和检查文件是否存在

url_base = 'http://yann.lecun.com/exdb/mnist/'
# MNIST官网有时不稳定，用这个镜像
url_base = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

filenames = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    # 训练集
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
    # 测试集
]

os.makedirs('../data', exist_ok=True)
# exist_ok=True是os.makedirs()的一个参数，意思是"如果文件夹已经存在，不要报错，直接跳过"。

for filename in filenames:
    url = url_base + filename
    # 拼接完整下载地址，比如
    # 'https://...mnist/' + 'train-images-idx3-ubyte.gz'
    save_path = os.path.join('../data', filename)
    # save_path = os.path.join('mnist_data', filename):
    # 拼的是本地路径，用来保存到你电脑上的哪个位置.
    # 对比 url = url_base + filename:
    # 拼的是网络地址，用来下载
    if not os.path.exists(save_path):
    # 检查这个文件本地是否已经存在，不存在才下载，存在就跳过——这样第二次运行不会重复下载。
        print(f'下载 {filename} ...')
        urllib.request.urlretrieve(url, save_path)
        # 从url下载文件，保存到save_path(就是我们创建的文件夹mnist_data的路径)
        print(f'完成')
    else:
        print(f'{filename} 已存在')

print('全部完成')




        # 加载和检查 MNIST #

import numpy as np
import gzip
# gzip是用来解压.gz文件的，MNIST下载下来是压缩格式，读取的时候需要先解压。
import os

def load_mnist():
    def load_images(filename):
        with gzip.open(os.path.join('../data', filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)  # 每张图片是28x28=784个像素
    # 这段是读取压缩文件的标准写法，属于"工具代码"，不需要深究。

    def load_labels(filename):
        with gzip.open(os.path.join('../data', filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    x_test  = load_images('t10k-images-idx3-ubyte.gz')
    y_test  = load_labels('t10k-labels-idx1-ubyte.gz')

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist()

print("训练集图片shape:", x_train.shape)  # 应该是 (60000, 784)
print("训练集标签shape:", y_train.shape)  # 应该是 (60000,)
print("测试集图片shape:", x_test.shape)   # 应该是 (10000, 784)
print("测试集标签shape:", y_test.shape)   # 应该是 (10000,)
print("第一张图片的标签（它是数字几）:", y_train[0])



        # 设置前向传播权重 #

import numpy as np
'''
numpy数组: (a,b) means a行b列的二维数组
输入              W1              输出
(1, 784)    ×  (784, 128)  =   (1, 128)

输入的具体含义(展平，我忘了)
一张28×28的图片
↓ 把所有行拼成一行(这样两个量就能既表达位置又表达灰度)
变成1×784的一行数字
每一个位置存的就是那个像素的灰度值

W1形状的含义:784个输入小像素灰值，128个输出节点数
'''
def init_params():
# 第一层：784个输入 → 128个节点
    W1 = np.random.randn(784, 128) * 0.01
    #np.random.randn(784, 128) 就是生成784×128个随机数，排列成784行128列
    '''
    随机抽出来的数可能很大，比如2.3、-1.8这种。
    如果w一开始就很大，第一次前向传播算出来的值会非常大，训练会不稳定。
    乘0.01就是把所有w都压缩到很小的范围，比如0.02、-0.018，让训练开头更平稳。
    超参数之一，是我们人为设定的。
    '''
    b1 = np.zeros(128)  # 生成 128个0，b的初始化值是0

# 第二层：128个节点 → 10个输出
    W2 = np.random.randn(128, 10) * 0.01
    b2 = np.zeros(10)
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()
print("W1的shape:", W1.shape)
print("W2的shape:", W2.shape)

#感受一下W1的实际具体内容和含义
print("W1里总共有多少个数:", W1.size)
print("W1的前3行第0列的值:", W1[:3, 0])  # 随机的，每次不一样




        # 前向传播 #

# 先做数据预处理
x_train = x_train / 255.0  # 归一化
x_test  = x_test  / 255.0

# 取第一张图片试试
x = x_train[0:1]  # shape是(1, 784)
print("输入shape:", x.shape)

# 过第一层
z1 = x @ W1 + b1    # *在numpy里是逐元素相乘，@是矩阵乘法，我们需要的是矩阵乘法
print("第一层输出shape:", z1.shape)  # 应该是(1, 128)

# 激活函数ReLU
def relu(z):
    return np.maximum(0, z)

a1 = relu(z1)
print("激活后shape:", a1.shape)  # 还是(1, 128)，shape不变


def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

z2 = a1 @ W2 + b2
a2 = softmax(z2)
print("第二层输出shape:", a2.shape)   # (1, 10)
print("输出的10个概率:", a2)
print("概率加起来等于:", np.sum(a2))  # 应该是1.0

    #总结一下前面做完的事：
    #输入(1,784) → @W1+b1 → ReLU → @W2+b2 → Softmax → 10个概率



        #下一步是计算损失，也就是"梯度"那个部分

def cross_entropy(a2, y):   # y是正确解标签
    return -np.log(a2[0, y] + 1e-7)  # 加1e-7防止log(0)

y = y_train[0]  # 第一张图的标签是5
loss = cross_entropy(a2, y)
print("当前损失:", loss)
print("正确类别(5)的概率:", a2[0, y])



        # 反向传播
def backward(x, y, a1, a2, z1):
    # 输出层的误差信号
    one_hot_y = np.zeros(10)
    one_hot_y[y] = 1
    dz2 = a2 - one_hot_y

    # 第二层梯度
    dW2 = a1.T @ dz2
    db2 = dz2

    # 传回第一层
    dz1 = (dz2 @ W2.T) * (z1 > 0)

# *(z1>0)的含义：
# (z1 > 0) 会生成一个和z1一样shape的矩阵，里面全是True或False，比如：
# z1        = [-0.5,  0.3, -0.1,  0.8]
# (z1 > 0)  = [False, True, False, True]
#           =  [0,    1,    0,    1]
# 所以* (z1 > 0)的意思就是：dz2往回传的信号，在z1小于等于0的位置全部清零，大于0的位置保留。

    # 第一层梯度
    dW1 = x.T @ dz1
    db1 = dz1

    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backward(x, y_train[0], a1, a2, z1)
print("dW1的shape:", dW1.shape)  # 应该是(784, 128)
print("dW2的shape:", dW2.shape)  # 应该是(128, 10)



        # 梯度下降，参数更新：批循环训练

# 训练参数
epochs = 10  # 把整个数据集过10遍
learning_rate = 0.01
batch_size = 32  # 每次用32张图更新一次参数

W1, b1, W2, b2 = init_params()  # 重新初始化

for epoch in range(epochs):
    # 打乱数据顺序
    indices = np.random.permutation(len(x_train))
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]

    total_loss = 0

    for i in range(0, len(x_train), batch_size):
        # 取一个batch
        x_batch = x_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # 前向传播
        z1 = x_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)

        # 计算损失
        batch_loss = cross_entropy(a2, y_batch)
        total_loss += batch_loss
        # 累加的原因 :
        # 因为一个 epoch 里数据是分成很多批次喂进去的，每个批次只算了一部分数据的 loss。累加起来之后除以批次数，就能得到整个 epoch 的平均 loss

        # 反向传播
        one_hot_y = np.zeros((len(y_batch), 10))
        # (len(y_batch), 10) 就是一个二维的行列数:
        # len(y_batch) → 行数，即这个批次有多少个样本
        # 10 → 列数，即 10 个类别（0-9 的数字）。10 是 MNIST 数据集的类别数——数字 0 到 9，刚好 10 个类别。
        for j in range(len(y_batch)):
            one_hot_y[j, y_batch[j]] = 1

        dz2 = (a2 - one_hot_y) / len(y_batch)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dz1 = (dz2 @ W2.T) * (z1 > 0)
        dW1 = x_batch.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # 更新参数
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

    # 每个epoch结束后测试准确率
    z1_test = x_test @ W1 + b1
    a1_test = relu(z1_test)
    z2_test = a1_test @ W2 + b2
    a2_test = softmax(z2_test)
    predictions = np.argmax(a2_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(x_train) * batch_size:.4f} | 测试准确率: {accuracy * 100:.2f}%")

# 输入(784) → @W1+b1 → z1 → relu → a1 → @W2+b2 → z2 → softmax → a2
# 数学计算得到的公式 dz2 = (a2 - one_hot_y) / len(y_batch)           onehot就用在这!



















# D2L--lesson5

## 线性回归

### **线性回归**

**房价预测**

![img](https://i0.hdslb.com/bfs/note/be06eb109c2cfdd202317d6e70e5321a98322d8c.png)

**线性模型**

![img](https://i0.hdslb.com/bfs/note/b1344eccb392f7600e161ca2d8270bbad0a02a6c.png)

- b：标量偏差

![img](https://i0.hdslb.com/bfs/note/44d4b09558558c6376ff006463c4d711b79d7f42.png)

- 单层神经网络：带权重的层只有一个（输出层）

**真实的神经元**

![img](https://i0.hdslb.com/bfs/note/72c9bbec794e831453267b13bc14b0bce6e16f82.png)

**衡量预估质量**

![img](https://i0.hdslb.com/bfs/note/41f71a1db6c370810d0cdd960a27730eb196732e.png)

- 平方损失：衡量真实值与预测值之间的误差
- 1 / 2：主要是为了求导的时候方便抵消平方的导数所产生的系数2

**训练数据**

![img](https://i0.hdslb.com/bfs/note/fd54b295441af971d2f81dcf4281b9d02bc2bda3.png)



**参数学习**

![img](https://i0.hdslb.com/bfs/note/415fee6cccf72f898a393c3c841dcefd3a152a26.png)

- 1 / 2：来自损失函数
- 1 / n：求平均
- 最小损失函数来学习参数



![img](https://i0.hdslb.com/bfs/note/ada8b449f2d3e5ab8aaadf1b1ce680be3456f47b.png)

- 因为是线性模型，所以是有显示解的，并且损失是一个凸函数
- 凸函数的最优解满足梯度等于0

**总结**

- 线性回归是对n维输入的加权，外加偏差（对输出值的预估）
- 使用平方损失来衡量预测值和真实值的差异
- 线性回归有显示解（一般来说，模型都没有显示解，有显示解的模型过于简单，复杂度有限，很难衡量复杂的数据）
- 线性回归可以看做是单层神经网络（最简单的神经网络）



### **梯度下降优化算法**

**梯度下降**

![img](https://i0.hdslb.com/bfs/note/4d6eb10d17454fd20a1fcf682728e085ea600d1c.png)

- η：标量，表示学习率，代表沿着负梯度方向一次走多远，即步长。他是一个超参数（需要人为的指定值）
- 学习率的选择不能太小，也不能太大（太小会导致计算量大，求解时间长；太大的话或导致函数值振荡，并没有真正的下降）

![img](https://i0.hdslb.com/bfs/note/fb5e2f1d31560248eb5dbec5551b3bda0733d4a6.png)

- l：损失函数
- 圆圈代表函数值的等高线，每个圆圈上的点处的函数值相等
- 梯度：使得函数值增加最快的方向，负梯度就是使函数下降最快的方向，如图中黄线所示

**小批量随机梯度下降**

![img](https://i0.hdslb.com/bfs/note/15a2cd20d2cd53ce59bb864e8c8be88dc472248c.png)

- 梯度下降时，每次计算梯度，要对整个损失函数求导，损失函数是对所有样本的平均损失，所以每求一次梯度，要对整个样的本的损失函数进行重新计算，计算量大且耗费时间长，代价太大
- 用 b 个样本的平均损失来近似所有样本的平均损失，当 b 足够大的时候，能够保证一定的精确度 
- 批量大小的选择

![img](https://i0.hdslb.com/bfs/note/3bef4f3e2e8bfa66b298a6741eece48b89a37ada.png)

**总结**



- 梯度下降通过不断沿着负梯度方向更新参数求解（好处是不需要知道显示解是什么，只需要不断的求导就可以了）
- 小批量随机梯度下降是深度学习默认的求解算法（虽然还有更稳定的，但是他是最稳定、最简单的）
- 两个重要的超参数是批量大小和学习率

### **线性回归的实现**

**1、导入所需要的包**

```python
%matplotlib inline
import random
import torch
from d2l import torch as d2l
```

- random：导入random包用于随机初始化权重
- d2l：将用过的或者实现过的算法放在d2l的包里面
- matplotlib inline：在plot的时候默认是嵌入到matplotlib中
- 报错 No module named 'matplotlib' ：在命令行中使用 pip install matplotlib 安装 matplotlib 包即可
- 报错 No module named 'd2l' ：在命令行中使用 pip install d2l 安装 d2l 包即可，安装完成之后可能需要重新打开程序才能生效

**2、根据带有噪声的线性模型构造一个人造的数据集**

![img](https://i0.hdslb.com/bfs/note/8f75323580776b9fa821bed2887f3237f4df9f3d.png)

- 构造人造数据集的好处是知道真实的 w 和 b 
- X = torch.normal(0,1,(num_examples,len(w)))：X 是一个均值为 0 ，方差为 1 的随机数，他的行数等于样本数，列数等于 w 的长度
- y += torch.normal(0,0.01,y.shape)：给 y 加上了一个均值为 0 ，方差为 0.01 形状和 y 相同的噪声
- return X,y.reshape((-1,1))：最后把 X 和 y 做成一个列向量返回
- true_w：真实的 w
- true_b：真实的 b
- features,labels = synthetic_data(true_w,true_b,1000)，根据函数来生成特征和标注

```python
def synthetic_data(w,b,num_examples):
  """生成 y = Xw + b + 噪声"""
  X = torch.normal(0,1,(num_examples,len(w)))
  y = torch.matmul(X,w) + b
  y += torch.normal(0,0.01,y.shape)
  return X,y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
```

**3、feature中每一行都包含一个二维数据样本，labels中的每一行都包含一维标签值**

```python
print('features:',features[0],'\nlabels',labels[0])
```

输出：

features: tensor([0.1605, 0.2305]) 

labels tensor([3.7450])

```python
d2l.set_figsize()
d2l.plt.scatter(features[:1].detach().numpy(),labels.detach().numpy(),1);
```

输出：

![img](https://i0.hdslb.com/bfs/note/036f1835fd08f18dfe7267e14d5f9c97986761e2.png)

- detach()：在pytorch的一些版本中，需要从计算图中detach出来才能转到numpy中去

**4、实现一个函数读取小批量**

定义一个data_iter函数，该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量

```python
def data_iter(batch_size,features,labels):
  num_examples = len(features)
  indices = list(range(num_examples))
  # 这些样本是随机读取的，没有特定顺序
  random.shuffle(indices)
  for i in range(0,num_examples,batch_size):
​    batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
​    yield features[batch_indices],labels[batch_indices]

batch_size = 10

for X,y in data_iter(batch_size,features,labels):
  print(X,'\n',y)
  break
```

- batch_size：批量大小
- num_examples：样本数
- random.shuffle(indices)：将下标打乱，实现对样本的随机访问
- for i in range(0,num_examples,batch_size)：从 0 开始到样本数结束，每次跳批量大小
- yield features[batch_indices],labels[batch_indices]：通过indices，每次产生随机顺序的特征和其对应的随即顺序标号
- yield是python中的一个迭代器

输出：

tensor([[-0.3785, -0.5235],

​        [ 0.0674,  0.6504],

​        [-0.9654,  1.3349],

​        [ 0.6241, -0.1228],

​        [-1.4924, -1.4087],

​        [ 0.7791, -0.5478],

​        [-0.6711, -0.5625],

​        [-0.0553, -0.1775],

​        [-0.7085,  0.6464],

​        [ 2.4625, -1.1113]]) 

 tensor([[ 5.2227],

​        [ 2.1381],

​        [-2.2681],

​        [ 5.8792],

​        [ 6.0258],

​        [ 7.6395],

​        [ 4.7813],

​        [ 4.6846],

​        [ 0.5918],

​        [12.8976]])

**5、定义初始化模型参数**

```python
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
```

**6、定义模型**

```python
def linreg(X,w,b):
  """线性回归模型"""
  return torch.matmul(X,w) + b
```

**7、定义损失函数**

```python
def squared_loss(y_hat,y):
  """均方损失"""
  return (y_hat - y.reshape(y_hat.shape))**2 / 2
```

- y_hat：预测值
- y：真实值
- 虽然 y_hat 和 y 元素个数是一样的，但是可能他们一个是行向量一个是列向量，因此需要使用reshape进行统一

**8、定义优化算法**

```python
def sgd(params,lr,batch_size):
  """小批量梯度下降"""
  with torch.no_grad():
    for param in params:
      param -= lr * param.grad / batch_size
      param.grad.zero_()
```

- params：给定的所有参数，包含 w 和 b ，他是一个list
- lr：学习率
- param.grad.zero_()：手动将梯度设置成 0 ，在下一次计算梯度的时候就不会和上一次相关了

**9、训练过程**

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epoch):
  for X,y in data_iter(batch_size,features,labels):
    l = loss(net(X,w,b),y) # X 和 y 的小批量损失
    # 因为 l 的形状是（batch_size,1），而不是一个标量。l 中的所有元素被加到一起求梯度
    # 并以此计算关于[w,b]的梯度
    l.sum().backward()
    sgd([w,b],lr,batch_size) # 使用参数的梯度更新
  with torch.no_grad():
    train_l = loss(net(features,w,b),labels)
    
    print(f'epoch{epoch + 1},loss{float(train_l.):f}')
```

- num_epoch=3：将整个数据扫描三遍
- net：之前定义的模型
- loss：均方损失
- 每一次对数据扫描一遍，扫描的时候拿出一定批量的X和y﻿

输出：

epoch1,loss0.034571

epoch2,loss0.000130

epoch3,loss0.000049

**10、比较真实参数和通过训练学到的参数来评估训练的成功程度**

```python
print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')
```

输出：

w的估计误差:tensor([ 0.0003, -0.0004], grad_fn=<SubBackward0>)

b的估计误差:tensor([-7.7724e-05], grad_fn=<RsubBackward1>)

### **线性回归的实现（简洁版）**

使用深度学习框架来简洁的实现线性回归模型生成数据集，使用pytorch的nn的moudule提供的一些数据预处理的模块来使实现更加简单

**1、导包**

```python
import numpy as np
import torch
import torch.utils.data as data
from d2l import torch as d2l

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = d2l.synthetic_data(true_w,true_b,1000)
```

- 这里沐神导包的时候写的是 import torch.utils as data ，我在运行的时候会报错 AttributeError: module 'torch.utils' has no attribute 'TensorDataset' ，当我改成 import torch.utils.data as data 这种方式来进行导包之后就能够继续运行了

**2、调用框架中现有的API来读取数据**

```python
def load_array(data_arrays,batch_size,is_train=True):
  """构造一个PyTorch的数据迭代器"""
  dataset = data.TensorDataset(*data_arrays)
  return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)

next(iter(data_iter))
```

输出：

[tensor([[-0.8249,  1.9358],

​         [-0.9235, -0.2573],

​         [-1.6639, -1.0040],

​         [ 0.5258,  1.1980],

​         [-0.4612, -0.8816],

​         [ 2.1845, -1.4419],

​         [-0.0317,  1.7666],

​         [-0.0510, -0.9682],

​         [-1.8175,  0.0168],

​         [ 0.5344,  0.5456]]),

 tensor([[-4.0278],

​         [ 3.2316],

​         [ 4.2774],

​         [ 1.1869],

​         [ 6.2739],

​         [13.4859],

​         [-1.8532],

​         [ 7.3907],

​         [ 0.5091],

​         [ 3.4219]])]

**3、使用框架的预定义好的层**

```python
from torch import nn
net = nn.Sequential(nn.Linear(2,1))
```

- nn 是神经网络的缩写
- nn.Linear(2,1)：输入的维度是2，输出的维度是1
- sequential：可以理解成是层的排列，将一层一层的神经网络进行排列在一起

**4、初始化模型参数**

```python
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
```

- normal_：使用正态分布来替换掉data的值（均值为0，方差为0.01）
- fill_：将偏差直接设置为0

输出：

tensor([0.])

**5、计算均方误差使用的是MESLoss类，也称为L2范数**

```python
loss = nn.MSELoss()
```

**6、实例化SGD实例**

```python
trainer = torch.optim.SGD(net.parameters(),lr=0.03)
```

**7、训练过程**

和之前的训练过程是一样的

```python
num_epochs = 3
for epoch in range(num_epochs):
  for X,y in data_iter:
    l = loss(net(X),y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
  l = loss(net(features),labels)
  print(f'epoch {epoch + 1}, loss {l:f}')
```

- trainer.step()：调用step函数来对模型进行更新

输出：

epoch 1, loss 0.000304

epoch 2, loss 0.000100

epoch 3, loss 0.000100
# D2L--lesson12 Pytorch神经网络基础

## 层和块

### 层和块是什么

*块*（block）可以描述单个层、由多个层组成的组件或整个模型本身。 使用块进行抽象的一个好处是可以将一些块组合成更大的组件， 这一过程通常是递归的。

块由*类*（class）表示。 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数， 并且必须存储任何必需的参数。

 最后，为了计算梯度，块必须具有反向传播函数。

下面的多层感知机的代码，包含一个具有256个单元和ReLU激活函数的全连接隐藏层， 然后是一个具有10个隐藏单元且不带激活函数的全连接输出层。：

```python
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

输出：

tensor([[-0.0317,  0.0599,  0.0677, -0.0397, -0.2779, -0.1492,  0.2559, -0.1335,
          0.1436,  0.0724],
        [ 0.0539,  0.0003,  0.2255,  0.1326, -0.1195, -0.2064,  0.2232, -0.1109,
          0.0944,  0.0557]], grad_fn=<AddmmBackward0>)

`nn.Sequential`定义了一种特殊的`Module`， 即在PyTorch中表示一个块的类， 它维护了一个由`Module`组成的有序列表。 

- `nn.Sequential` 和 `nn.Module` 可以嵌套使用

### 自定义块

1. 将输入数据作为其前向传播函数的参数。
2. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
3. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
4. 存储和访问前向传播计算所需的参数。
5. 根据需要初始化模型参数。

实现只需要提供我们自己的构造函数（Python中的`__init__`函数）和前向传播函数。

```python
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))
```

块的一个主要优点是它的多功能性。 我们可以子类化块以创建层（如全连接层的类）、 整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。

### 顺序块

`Sequential`的设计是为了把其他模块串起来。 构建自己的简化的`MySequential`， 只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数。
2. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

使用我们的`MySequential`类重新实现多层感知机:

```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

输出：

tensor([[-0.1438, -0.0631, -0.2420, -0.0499,  0.2361,  0.0188,  0.1304,  0.2161,
         -0.1725, -0.0187],
        [-0.1461, -0.0612, -0.1650,  0.0629,  0.3020, -0.0028,  0.1891,  0.1974,
         -0.2368, -0.2183]], grad_fn=<AddmmBackward0>)

### 在前向传播函数中执行代码

`Sequential`类使模型构造变得简单， 允许我们组合新的架构，而不必定义自己的类。 然而，并不是所有的架构都是简单的顺序架构。 当需要更强的灵活性时，我们需要定义自己的块。 

```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。 这个权重不是一个模型参数，因此它永远不会被反向传播更新。 然后，神经网络将这个固定层的输出通过一个全连接层。

也可以混合搭配：

```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

## 参数管理

### 参数访问

通过索引来访问模型的任意层。

```python
print(net[2].state_dict())
```

输出：

OrderedDict([('weight', tensor([[ 0.3231, -0.3373,  0.1639, -0.3125,  0.0527, -0.2957,  0.0192,  0.0039]])), ('bias', tensor([-0.2930]))])

* 全连接层包含两个参数，分别是该层的权重和偏置。

访问目标参数：

```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

输出：

```python
<class 'torch.nn.parameter.Parameter'>
Parameter containing:
tensor([-0.2930], requires_grad=True)
tensor([-0.2930])
```

也可以一次性的访问一层或所有层的参数：

```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

输出：

('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))

另一种访问网络参数的方法：

```python
net.state_dict()['2.bias'].data
```

输出：

tensor([-0.2930])

从嵌套块收集参数：

```python
#定义嵌套块：
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

#打印网络结构：
print(rgnet)
```

输出：

Sequential(
  (0): Sequential(
    (block 0): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 1): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 2): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
    (block 3): Sequential(
      (0): Linear(in_features=4, out_features=8, bias=True)
      (1): ReLU()
      (2): Linear(in_features=8, out_features=4, bias=True)
      (3): ReLU()
    )
  )
  (1): Linear(in_features=4, out_features=1, bias=True)
)

访问方法：

```python
#第一个主要的块中、第二个子块的第一层的偏置项。
rgnet[0][1][0].bias.data
```

输出：

tensor([-0.2726,  0.2247, -0.3964,  0.3576, -0.2231,  0.1649, -0.1170, -0.3014])

### 参数初始化

**内置初始化器**

所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0：

```python
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

输出：

(tensor([-0.0017,  0.0232, -0.0026,  0.0026]), tensor(0.))

初始化为给定的常数，比如初始化为1：

```python
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

输出：

(tensor([1., 1., 1., 1.]), tensor(0.))

使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42：

```python
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

输出:

tensor([-0.4645,  0.0062, -0.5186,  0.3513])
tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])

**自定义初始化器**

```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

Init weight torch.Size([8, 4])
Init weight torch.Size([1, 8])
tensor([[ 8.8025,  6.4078,  0.0000, -8.4598],
        [-0.0000,  9.0582,  8.8258,  7.4997]], grad_fn=<SliceBackward0>)

### 参数绑定

在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。

```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

输出：

tensor([True, True, True, True, True, True, True, True])
tensor([True, True, True, True, True, True, True, True])

第三个和第五个神经网络层的参数是绑定的。 它们不仅值相等，而且由相同的张量表示。 因此，如果我们改变其中一个参数，另一个参数也会改变。 

当参数绑定时，梯度会发生什么情况？

由于模型参数包含梯度，因此在反向传播期间第二个隐藏层 （即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。

## 自定义层

### 不带参数的层

下面的`CenteredLayer`类要从其输入中减去均值。 要构建它，我们只需继承基础层类并实现前向传播功能。

```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```python
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

输出：

tensor([-2., -1.,  0.,  1.,  2.])

将定义的层插入到现有的网络：

```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
```

输出：

tensor(9.3132e-10, grad_fn=<MeanBackward0>)

### 带参数的层

实现自定义版本的全连接层。 回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。 在此实现中，我们使用修正线性单元作为激活函数。

```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```python
#实例化
linear = MyLinear(5, 3)
linear.weight
```

输出：

Parameter containing:
tensor([[ 1.9054, -3.4102, -0.9792],
        [ 1.5522,  0.8707,  0.6481],
        [ 1.0974,  0.2568,  0.4034],
        [ 0.1416, -1.1389,  0.5875],
        [-0.7209,  0.4432,  0.1222]], requires_grad=True)

直接进行前向传播：

```python
linear(torch.rand(2, 5))
```

输出：

tensor([[2.4784, 0.0000, 0.8991],
        [3.6132, 0.0000, 1.1160]])

也可以使用自定义层构建模型：

```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

输出：

tensor([[0.],
        [0.]])

## 读写文件

### 加载和保存张量

直接调用`load`和`save`函数分别读写它们。

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

将存储在文件中的数据读回内存

```python
x2 = torch.load('x-file')
x2
```

输出：

tensor([0, 1, 2, 3])

也可以存储一个张量列表，然后把它们读回内存

```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

输出：

(tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))

甚至写入或读取从字符串映射到张量的字典

```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

输出：

{'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}

### 加载和保存模型参数

深度学习框架提供了内置函数来保存和加载整个网络。 需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。

以下面的MLP为例：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

将模型的参数存储在一个叫做“mlp.params”的文件中

```python
torch.save(net.state_dict(), 'mlp.params')
```

为了恢复模型，我们实例化了原始多层感知机模型的一个备份

```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

输出：

MLP(
  (hidden): Linear(in_features=20, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)

验证一下：

```python
Y_clone = clone(X)
Y_clone == Y
```

输出：

tensor([[True, True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True, True]])

## GPU

### 使用GPU进行计算

首先，确保你至少安装了一个NVIDIA GPU。 然后，下载[NVIDIA驱动和CUDA](https://developer.nvidia.com/cuda-downloads) 并按照提示设置适当的路径。 当这些准备工作完成，就可以使用`nvidia-smi`命令来查看显卡信息。

```python
!nvidia-smi
```

默认情况下，所有变量和相关的计算都分配给CPU，当在带有GPU的服务器上训练神经网络时， 我们通常希望模型的参数在GPU上

在PyTorch中，CPU和GPU可以用`torch.device('cpu')` 和`torch.device('cuda')`表示。

 应该注意的是，`cpu`设备意味着所有物理CPU和内存， 这意味着PyTorch的计算将尝试使用所有CPU核心。 然而，`gpu`设备只代表一个卡和相应的显存。 如果有多个GPU，我们使用`torch.device(f'cuda:{i}')` 来表示第\(i\)块GPU（\(i\)从0开始）。 另外，`cuda:0`和`cuda`是等价的。

```python
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

输出：

(device(type='cpu'), device(type='cuda'), device(type='cuda', index=1))

查询可用gpu的数量：

```python
torch.cuda.device_count()
```

下面定义两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。

```python
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

输出：

(device(type='cuda', index=0),
 device(type='cpu'),
 [device(type='cuda', index=0), device(type='cuda', index=1)])

### 张量与GPU

默认情况下，张量是在CPU上创建的。

无论何时我们要对多个项进行操作， 它们都必须在同一个设备上。

```python
X = torch.ones(2, 3, device=try_gpu())
Y = torch.rand(2, 3, device=try_gpu(1))
#此时不能执行X+Y,因为X，Y在不同的GPU上
Z = X.cuda(1) #将X复制一份到第二块GPU上
Y+Z #现在执行加操作
```

输出：

tensor([[1.1206, 1.2283, 1.4548],
        [1.9806, 1.9616, 1.0501]], device='cuda:1')

### 神经网络与GPU

```python
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

当输入为GPU上的张量时，模型将在同一GPU上计算结果

只要所有的数据和参数都在同一个设备上， 我们就可以有效地学习模型
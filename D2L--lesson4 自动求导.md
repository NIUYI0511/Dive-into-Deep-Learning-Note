# D2L--lesson-4

## 自动求导

### **向量链式法则**

![img](https://i0.hdslb.com/bfs/note/be670069e3f13c5ec551947f911a23e43d2ce5ab.png)

**例1：**

![img](https://i0.hdslb.com/bfs/note/33014324e15ff2c9f534aad2c1e1703a83de3ea8.png)

- 线性回归的例子：利用链式法则对线性回归求导
- <x,w>：x 和 w 做内积
- 上图中求 z 对 w 的导数时，第二项 b 对 a 的导数在代入时应该应该写作 d (a - y)  ，图中少了一对括号

**例2**：

![img](https://i0.hdslb.com/bfs/note/39ff59819cf48a3abeac0c48e317f72a49fa0de1.png)

- 上图中求 z 对 w 的导数时，第二项 b 对 a 的导数在代入时应该应该写作 d (a - y)  ，图中少了一对括号
- 图中 ||b|| 表示求 b 的范数

### **自动求导**

计算一个函数在指定值上的导数

自动求导与符号求导、数值求导的区别

![img](https://i0.hdslb.com/bfs/note/a28226d4d1598e2da569fae05c4b60fe4ccf7974.png)

- 符号求导：指定一个函数，然后求函数的导数（这里可以理解为对某一个函数进行求导，求出其导函数，而不是一个具体的值），它是一个显式的计算
- 数值求导：给定一个未知函数（不知道其具体表达式），然后利用数值去拟合它的导数（这里有点像利用极限去求导的过程）

### **计算图**

虽然使用pytorch不用去理解计算图，但是有必要知道其内部的工作原理，便于使用其它框架（如tensorflow等）时去理解计算图

![img](https://i0.hdslb.com/bfs/note/bbc3e17339e4f376e987df877807e74e0b1befec.png)

- 加入中间变量 a 、 b ，将他们做成一个基本的计算子
- 每个圆圈表示一个操作，也可以表示成为一个输入

显式构造

（Tensorflow / Theano / MXNet）

```python
from mxnet import sym
a = sym.var()
b = sym.var()
c = 2 * a + b
# bind data into a and b later
```

- 此处会报错“No module named 'mxnet'”，直接在conda环境下使用 pip install mxnet 或者沐神之前用的在程序第一行使用 !pip install mxnet 这行代码进行安装也可

输出：

TypeError: var() missing 1 required positional argument: 'name'

- 这里会提示缺少一个位置参数
- 这里应该不是一段完整的可运行的代码，只是为了介绍显式构造

### 隐式构造

（PyTorch / MXNet）

```python
from mxnet import autograd , nd
with autograd.record():
  a = nd.ones((2,1))
  b = nd.ones((2,1))
  c = 2 * a + b
```

输出：

- 没有输出，但也不会报错

### **自动求导的两种模式**

![img](https://i0.hdslb.com/bfs/note/87143aa678a3fd2c9c58c2ec0a81ecf3eca23413.png)

反向积累

![img](https://i0.hdslb.com/bfs/note/5cc0101bcfa5df40d5e68a499c839216dfda7184.png)

![img](https://i0.hdslb.com/bfs/note/248c2f3f0030c99824cea5130f5dd2a6d2f50e0f.png)

反向累积总结

![img](https://i0.hdslb.com/bfs/note/47ce3480f49da9f9c65d5a46ab88a21eb7dc7bff.png)

### **复杂度**

![img](https://i0.hdslb.com/bfs/note/ead1f1b649220a397848b100747c48fdebb969d7.png)

- 内存复杂度：深度神经网络耗费GPU资源的根源，做梯度的时候需要存储前面的运行结果
- 正向累积：不管神经网络的深度有多深都不需要存储任何的结果，但是因为需要对每一层计算梯度，所以每计算一个梯度都需要重新扫描一遍，计算复杂度太高

### **自动求导的实现**

![img](https://i0.hdslb.com/bfs/note/6c884b45d5cdaee4d81a64552cead3a94aee756e.png)

```python
import torch
x = torch.arange(4.0)
x
```

输出：

tensor([0., 1., 2., 3.])

在计算 y 关于 向量x 的梯度之前，需要一个地方来存储梯度

```python
x.requires_grad_(True)
# 等价于 x = torch.arange(4.0，requires_grad=True)
x.grad # 默认值是None
```

然后计算y

```python
y = 2 * torch.dot(x,x)
y
```

- y 等于 x 与 x 的内积的两倍

输出：

tensor(28., grad_fn=<DotBackward0>)

- grad_fn：这里是隐式构造图，所以有一个求梯度的函数，告知 y 是从 x 计算过来的

通过调用反向传播函数来自动计算 y 关于 x 每个分量的梯度

```python
y.backward()
x.grad
```

- y.backward()：求导
- 利用 x.grad 来访问导数

输出：

```python
tensor([ 0.,  4.,  8., 12.])
x.grad == 4 * x
```

- 验证一下结果

输出：

tensor([True, True, True, True])

计算 x 的另一个函数

```python
# 在默认情况下，PyTorch会累积梯度，因此在进行另外一个函数之前需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

- PyTorch中下划线表示重写内容，zero_表示把所有的梯度清零

输出：

tensor([1., 1., 1., 1.])

深度学习中，目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数之和

- 深度学习中很少对一个向量进行求导，而只是对一个标量进行求导
- 绝大部分情况下会先进行求和，再进行求导，这样子就是一个标量

```python
# 对非标量调用 backward 需要传入一个 gradient 参数
x.grad.zero_()
y = x * x
# 等价于 y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad
```

- y.detach()：把 y 当作一个常数而不是一个关于 x 的函数

输出;

tensor([0., 2., 4., 6.])

- 以后将会用于在一些场景下将网络中的参数固定

即使构建函数的计算图需要通过Python控制流（例如：条件、循环或任意函数调用），仍然可以计算得到变量的梯度

```python
def f(a):
  b = a * 2
  while b.norm() < 1000:
        b = b * 2
  if b.sum() > 0:
        c = b
  else:
        c = 100 * b
  return c

a = torch.randn(size = (),requires_grad=True)
d = f(a)
d.backward()
```

- size=()：表示是一个标量
- require_grad=True：表示需要梯度
- 在计算的时候，torch会把整个计算图存下来，然后倒着做一遍
- 隐式计算的好处在于对于控制流做的更好一些，但是速度比较慢
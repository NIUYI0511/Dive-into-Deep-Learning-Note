# D2L--lesson6

## softmax回归

### **softmax回归**

![img](https://i0.hdslb.com/bfs/note/7423ca3722f619cd8c3f9611916c7ea33a913892.png)

- softmax回归是机器学习中非常重要而且经典的模型，他的名字虽然叫回归，但实际上是一个分类问题

### **分类与回归**

- 回归是估计一个连续值
- 分类是预测一个连续的类别
- 分类问题举例

![img](https://i0.hdslb.com/bfs/note/390ea7f814700963fb77e30b71a2014777a88d80.png)

![img](https://i0.hdslb.com/bfs/note/de36405ac1ca7952522ebf755518d70ee7235447.png)

![img](https://i0.hdslb.com/bfs/note/25d4ac6e71641110aca2838e5ab03a1ba466e4f7.png)

![img](https://i0.hdslb.com/bfs/note/b8a74e98cbd0c61be795e05f496f1f8011b91c4f.png)

### **从回归到多类分类**

![img](https://i0.hdslb.com/bfs/note/c71e9479ba33b2d6ad5739dea616ad30571d3731.png)

- 分类问题从单输出变成了多输出，输出的个数等于类别的个数

![img](https://i0.hdslb.com/bfs/note/980f76aeb916f8fbe60889f6379d1cb0e8d996ee.png)

- 类别是一个数，也可能是一个字符串
- 一位有效编码：刚好有一个位置有效的编码方式（有效的那一位为 1 ，其它位上全部置 0 ）

![img](https://i0.hdslb.com/bfs/note/438a37d27df62b723102a02e766d5ae10578cdf7.png)

- 不关心置信度的值是多少，只关心正确类别的置信度的值要远远高于其他非正确类置信度，使得正确的类别能够与其他类别拉开距离
- 虽然关心的是一个相对值，但是需要把这些值放在一个合适的区间

![img](https://i0.hdslb.com/bfs/note/762ea6f76120c43f0a16b9d3069d5e4a6d435df7.png)

- 希望是输出能够尽量是一个概率
- y^：一个长为 n 的向量，每个元素都非负，而且这些元素加起来的和为 1
- exp（o）：e 的 o 次方，e 是一个指数常数。指数的好处是：不管是什么值都能将它变成非负
- 通过 softmax 将输出 o 变成了概率

### softmax交叉熵损失

![img](https://i0.hdslb.com/bfs/note/2c1db613e6678b3a1d605091b8888ba79e6faffc.png)

- 一般来说使用真实概率与预测概率的区别来作为损失
- 假设 p 和 q 是两个离散概率
- 上图中第二式的化简：根据前面的一位有效编码，yi中只有一位有效元素为1，其他都为0
- 对分类问题来讲，不关心对于非正确类的预测值，只关心对于正确类的预测值及其置信度的大小

**总结**

- softmax 回归是一个多分类分类模型
- 使用 softmax 操作子得到每个类的预测置信度（使得每个类的置信度是一个概率，非负且和为1）
- 使用交叉熵来衡量预测和标号的区别（作为损失函数）

### **损失函数**

![img](https://i0.hdslb.com/bfs/note/5bebce0fa0f4875b72b172e640e033af30495916.png)

- 损失函数用来衡量预测值和真实值之间的区别

**常用的损失函数**

1、L2 Loss（均方损失）

![img](https://i0.hdslb.com/bfs/note/ffacd6ea2cb4beddfab392dac64e7a236eee0c0d.png)

- 蓝色曲线表示 y = 0 的时候变化预测值 y‘ 的函数，这是一个二次函数
- 绿色曲线是它的似然函数，即exp（ - l ），它的似然函数是一个高斯分布
- 橙色曲线表示损失函数的梯度，它是一条过原点的直线
- 在梯度下降的时候是根据负梯度方向来更新梯度，所以它的导数决定如何更新参数

![img](https://i0.hdslb.com/bfs/note/ccfc12928e1cb64b7928a547d23bd3d251696821.png)

- 当 y 和 y‘ 离得比较远（横轴到零点的距离越远），梯度越大，对参数的更新越多，更新的幅度越大，反之亦然

2、L2 Loss（绝对值损失函数）

![img](https://i0.hdslb.com/bfs/note/a7a0949a17aff03fdc81845bebec7557eda3afe3.png)

- 蓝色曲线表示 y = 0 的时候变化预测值 y‘ 的函数
- 绿色曲线是它的似然函数，即exp（ - l ）
- 橙色曲线表示损失函数的梯度，当 距离 > 1 的时候他的值为1，当 距离 < 1 的时候他的值为-1
- 绝对值函数在0点处不可导
- 在梯度下降的时候是根据负梯度方向来更新梯度，它的梯度永远是常数，所以就算 y 和 y‘ 离得比较远，参数更新的幅度也不会太大，会带来稳定性上的好处
- 它的缺点是在 0 点处不可导，另外在 0 点处有一个 -1 到 1 的剧烈变化，不平滑性导致预测值和真实值靠的比较近的时候，也就是优化到了末期的时候这个地方可能就变得不那么稳定

Huber's Robust Loss（Huber鲁棒损失）

![img](https://i0.hdslb.com/bfs/note/1ef22b973141feebb11e42ba05514f003d770ae3.png)

- 当真实值和预测值相差较大时，他是一个绝对误差，而当他们相差较小时是一个均方误差
- 减去 1 / 2 的作用是使得分段函数的曲线能够连起来
- 蓝色曲线表示 y = 0 的时候变化预测值 y‘ 的函数，在 -1 到 1 之间是一个比较平滑的二次函数，在这个区间外是一条直线
- 绿色曲线是它的似然函数，即exp（ - l ），和高斯分布有点类似，但是不想绝对值误差那样在 0 点处有一个很尖的地方
- 橙色曲线表示损失函数的梯度，当 距离 > 1 的时候他的值为 -1 或者 1，是一个常数，当距离 < 1 的时候是一个渐变的过程，这样的好处是当预测值和真实值差得比较远的时候，参数值的更新比较均匀，而当预测值和真实值相差比较近的时候，即在参数更新的末期，梯度会越来越小，保证优化过程是比较平滑的，不会出现数值上的剧烈变化

### **图片分类数据集**

MNIST数据集（对手写数字的识别）是图像分类中广泛使用的数据集之一，但是作为基准数据集过于简单，因此使用类似但是更复杂的Fashion-MNIST数据集

**1、导包**

```python
%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
d2l.use_svg_display()
```

- torchvision：pytorch对于机器学习实现的一个库
- from torch.utils import data：方便读取数据的一些小批量的函数
- transforms：对数据进行操作的模块
- from d2l import torch as d2l：将一些函数实现好之后存在 d2l 中
- d2l.use_svg_display()：使用 svg 来显示图片，清晰度会更高一些



**2、通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中**

```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=trans,download=True)

len(mnist_train),len(mnist_test)
```

- trans = transforms.ToTensor()：将图片转换成pytorch中的tensor，通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在 0 到 1 之间
- train=True：代表下载的是训练集数据集
- download=True：可以将数据集加载好然后放在指定的目录下就可以不用download了
- train=False：代表下载的是测试集数据集
- 报错：<urlopen error [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。>，科学上网之后可以解决这个问题

输出：

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

Using downloaded and verified file: ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz

Extracting ../data\FashionMNIST\raw\train-images-idx3-ubyte.gz to ../data\FashionMNIST\raw



Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

Using downloaded and verified file: ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz

Extracting ../data\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw



Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz

100.0%

Extracting ../data\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ../data\FashionMNIST\raw



Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz

119.3%

Extracting ../data\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ../data\FashionMNIST\raw



(60000, 10000)

```python
mnist_train[0][0].shape
```

输出：

torch.Size([1, 28, 28])

- torch.Size([1, 28, 28])：因为它是一个黑白图片，所以channel数是 1

**3、定义两个可视化数据集的函数**

```python
def get_fashion_mnist_labels(labels):  #@save
  """返回Fashion-MNIST数据集的文本标签。"""
  text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
  return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
  """Plot a list of images."""
  figsize = (num_cols * scale, num_rows * scale)
  _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    if torch.is_tensor(img):
      # 图片张量
      ax.imshow(img.numpy())
    else:
      # PIL图片
      ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  return axes
```



**4、几个样本的图像及其相应的标签**

```python
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

- next：拿到第一个小批量

输出：

![img](https://i0.hdslb.com/bfs/note/f9c650b690bf5ea42e590c8db43561344700c761.png)

- X.reshape(18, 28, 28)：18张28 * 28的图片
- 2, 9：2 行 9 列
- titles=get_fashion_mnist_labels(y)：图片的名称显示其对应的标号

**5、读取一小批量数据，大小为batch_size**

```python
batch_size = 256

def get_dataloader_workers():  #@save
  """使用4个进程来读取数据。"""
  return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
  continue
f'{timer.stop():.2f} sec'
```

- 定义一个函数 get_dataloader_workers() 每次返回 4 ，使用 4 个进程数来进行读取，可以根据 cpu 的大小来寻阿泽一个小一点的数或者大一点的数
- shuffle=True：指定是否打乱顺序
- num_workers=get_dataloader_workers()：分配多少个进程，这里给定的是 4 
- timer = d2l.Timer()：timer 用来测试速度

输出：

'4.00 sec'

- 读取一次数据的时间是 4 秒
- 经常会遇到一个性能问题：模型训练速度比较快，但是数据读不过来，训练之前可以看一下数据读取的速度，一般要比模型的训练速度要快，这是一个常见瓶颈

**6、整合 load_data_fashion_mnist() 函数**

```python
def load_data_fashion_mnist(batch_size, resize=None):  #@save
  """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
  mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
  return (data.DataLoader(mnist_train, batch_size, shuffle=True,
              num_workers=get_dataloader_workers()),
      data.DataLoader(mnist_test, batch_size, shuffle=False,
              num_workers=get_dataloader_workers()))
```



- resize：是否将图片变得更大，后续可能需要更大的图片

### **softmax回归的从零开始实现**

**1、导包**

```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

- batch_size = 256：每次随机读取256张图片
- d2l.load_data_fashion_mnist(batch_size)：调用存储在 d2l 中的读取数据集的函数（上面已经实现了）
- train_iter, test_iter：返回训练集和测试集的迭代器

**2、展平每个图像，将他们视为长度为 784 的向量，因为数据集有 10 个类别，所以网络的输出维度是 10**

```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

- 每张图片都是 28 *28，通道数为 1.是一个三维的输入，但是对于 softmax 回归来讲，它的输入需要是一个向量，所以需要将图片拉长成为一个向量
- 这个“拉长”的操作会损失很多的空间信息，这里留给卷积神经网络来提取特征信息
- W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)：定义权重 W，将他初始为一个高斯随机分布的值，均值为 0，方差为 0.01.它的形状是行数等于输入的个数，列数等于输出的个数，因为要计算梯度，所以requires_grad=True
- b = torch.zeros(num_outputs, requires_grad=True)：定义偏移量 b：它的长度等于输出的个数，因为要计算梯度，所以requires_grad=True

**3、回顾：给定一个矩阵 X ，对所有元素求和**

```python
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

X.sum(0, keepdim=True), X.sum(1, keepdim=True)
```

- 0：将 shape 中的第 0 个元素变成 1，这就变成了一个行向量
- 1：将 shape 中的第 1 个元素变成 1，这就变成了一个列向量

输出：

(tensor([[5., 7., 9.]]),

 tensor([[ 6.],[15.]]))

**4、实现 softmax**

![img](https://i0.hdslb.com/bfs/note/58e086809cecf6825a036fe6fbe0773a5e1a6bfb.png)

```python
def softmax(X):
  X_exp = torch.exp(X)
  partition = X_exp.sum(1, keepdim=True)
  return X_exp / partition  # 这里应用了广播机制
```

- torch.exp(X)：将 X 的每一个元素做指数运算
- X_exp.sum(1, keepdim=True)：按照 shape 中的第一个元素求和，即按照每一行来进行求和

**5、将每一个元素变成一个非负数。根据概率原理，每行的总和为 1** 

```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
```

- X = torch.normal(0, 1, (2, 5))：创建一个均值为 0，方差为 1 的2行5列的矩阵 X

输出：

(tensor([[0.3740, 0.1609, 0.1182, 0.0572, 0.2897],

​         [0.2156, 0.2721, 0.1700, 0.3240, 0.0184]]),

 tensor([1., 1.]))

**6、实现 softmax 回归模型**

```python
def net(X):
  return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

**7、实现交叉熵损失**

补充：创建一个数据 y_hat，其中包含 2 个样本在 3 个类别的预测概率，使用 y 作为 y_hat 的索引

```python
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

- y_hat[[0, 1], y]给定一个预测值，然后拿出给定标号的真实类别的预测值是多少。对应[0][0]，[1][2]

输出：

tensor([0.1000, 0.5000])

**8、实现交叉熵损失函数**

```python
def cross_entropy(y_hat, y):
  return - torch.log(y_hat[range(len(y_hat)), y])
cross_entropy(y_hat, y)
```

输出：

tensor([2.3026, 0.6931])

**9、将预测类别和真实 y 元素进行比较**

```python
def accuracy(y_hat, y):  #@save
  """计算预测正确的数量。"""
  if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
​    y_hat = y_hat.argmax(axis=1)
  cmp = y_hat.type(y.dtype) == y
  return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)
```

输出：

0.5

**10、评估在任意模型 net 的准确率**

```python
def evaluate_accuracy(net, data_iter):  #@save
  """计算在指定数据集上模型的精度。"""
  if isinstance(net, torch.nn.Module):
    net.eval()  # 将模型设置为评估模式
  metric = Accumulator(2)  # 正确预测数、预测总数
  for X, y in data_iter:
    metric.add(accuracy(net(X), y), y.numel())
  return metric[0] / metric[1]
```

**11、Accumulator实例中创建了 2 个变量，用于分别储存正确预测的数量和预测的总数量**

```python
class Accumulator:  #@save
  """在`n`个变量上累加。"""
  def __init__(self, n):
    self.data = [0.0] * n

  def add(self, *args):
    self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
    self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]

evaluate_accuracy(net, test_iter)
```

输出：

0.1274

**12、softmax回归的训练**

我们定义⼀个函数来训练⼀个迭代周期。请注意，updater是更 新模型参数的常⽤函数，它接受批量⼤⼩作为参数。它可以是d2l.sgd函数，也可以是框架的内置优化函数。

```python
def train_epoch_ch3(net, train_iter, loss, updater): #@save
	"""训练模型⼀个迭代周期（定义⻅第3章）"""
	# 将模型设置为训练模式
	if isinstance(net, torch.nn.Module):
		net.train()
	# 训练损失总和、训练准确度总和、样本数
	metric = Accumulator(3)
	for X, y in train_iter:
	# 计算梯度并更新参数
		y_hat = net(X)
		l = loss(y_hat, y)
	if isinstance(updater, torch.optim.Optimizer):
		# 使⽤PyTorch内置的优化器和损失函数
	updater.zero_grad()
	l.mean().backward()
	updater.step()
	else:
	# 使⽤定制的优化器和损失函数
		l.sum().backward()
		updater(X.shape[0])
		metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
	# 返回训练损失和训练精度
	return metric[0] / metric[2], metric[1] / metric[2]

```

实现⼀个训练函数，它会在train_iter访问到的训练数据集上训练⼀个模型net。该训练函数 将会运⾏多个迭代周期（由num_epochs指定）。在每个迭代周期结束时，利⽤test_iter访问到的测试数 据集对模型进⾏评估。我们将利⽤Animator类来可视化训练进度。

```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
	"""训练模型（定义⻅第3章）"""
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
	train_loss, train_acc = train_metrics	
	assert train_loss < 0.5, train_loss
	assert train_acc <= 1 and train_acc > 0.7, train_acc
	assert test_acc <= 1 and test_acc > 0.7, test_acc
```

使⽤⼩批量随机梯度下降来优化模型的损失函数，设置学习 率为0.1。

```python
lr = 0.1
def updater(batch_size):
	return d2l.sgd([W, b], lr, batch_size)
```

训练模型10个迭代周期。

```python
num_epochs = 10 
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

13、softmax 预测

```python
def predict_ch3(net, test_iter, n=6): #@save
	"""预测标签（定义⻅第3章）"""
	for X, y in test_iter:
		break
	trues = d2l.get_fashion_mnist_labels(y)
	preds = 			d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
	titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
	d2l.show_images(
		X[0:n].reshape((n, 28, 28)), 1, n, 	titles=titles[0:n])
    
predict_ch3(net, test_iter)
```

小结:

* 借助softmax 回归，我们可以训练多分类的模型。 

* 训练softmax 回归循环模型与训练线性回归模型⾮常相似：先读取数据，再定义模型和损失函数，然后 使⽤优化算法训练模型。⼤多数常⻅的深度学习模型都有类似的训练过程。

### softmax 简洁实现

```python
# PyTorch不会隐式地调整输⼊的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整⽹络输⼊的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```


# D2L-lesson1

## **数据操作和数据预处理**

### **N维数组**

- 机器学习中使用最多的是**N维数组**

![img](https://i0.hdslb.com/bfs/note/4fcc4c330f852170fb6ba65ff7dd452fd655027f.png)

![img](https://i0.hdslb.com/bfs/note/dadc499d129ccb934c53967fc10bcd69ced682ca.png)

- 最简单的N为数组是**0维数组**，称之为**标量**，可能表示**一个物体的类别**
- 一维数组称为向量，比如说特征向量（将一个样本抽象成一行数字）
- 二维数组称为矩阵，比如说特征矩阵（行表示样本，列表示特征）
- 三维数组最简单的就是一张RGB图片（宽 （列数）* 高（行数） * 通道数（R、G、B））
- n个三维数组放在一块就变成了四维数组，比如说一个RGB图片的批量（batch，深度学习中一个batch中通常会有128张图片）
- 五维数组举例：一个视频的批量（在图片批量的基础上新增了时间的维度：批量大小 * 时间 * 宽 * 高 * 通道）

### **创建数组**

创建数组需要：

- 形状：矩阵的尺寸，比如3 * 4
- 每个元素的数据类型：比如32位浮点数
- 每个元素的值：例如全是0或者随机数（正态分布、均匀分布）

![img](https://i0.hdslb.com/bfs/note/a26b5a9865e12387e3b5d43578b012b264818643.png)



### **访问元素**

![img](https://i0.hdslb.com/bfs/note/5afb0b516b60d487c286ea0fa7f0f9a12b189b48.png)

- 元素行和列的下标都是从0开始的
- 上图中第三个例子有误，一列应该是[：，1]
- [1：3，1：]：“：”后面没有数字的话表示取到行或者列的末尾
- [：：3，：：2]：从第零行到最后一行，每三行一跳；从第零列到最后一列，每两列一跳



### **数据操作**

**1、导入torch（注意：虽然它被称为PyTorch，但是实际上应该导入Torch）**

```python
import torch
```

此处如果报错“no module named torch”，可以到pytorch的官网（直接百度pytorch就能找到pytorch的官网）复制pytorch的安装命令进行安装

**2、张量表示一个数值组成的数组，这个数组可能有多个维度**

```python
x = torch.arange(12)
x
```

- arange(12)：0 - 12 所有的数，不包含12(左闭右开)

输出：

tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

**3、可以通过shape属性来访问张量的形状和张量中元素的总数**

```python
x.shape
```

- 注意shape是属性，所以没有括号

输出：

torch.Size([12])

```python
x.numel()
```

- number of elements

输出：

12

**4、要改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数**

```python
x = x.reshape(3,4)
x
```

- 转换成三行四列的矩阵

输出：

tensor([[ 0,  1,  2,  3],

​        [ 4,  5,  6,  7],

​        [ 8,  9, 10, 11]])

**5、使用全0、全1、其他常量或者从特定分布中随机采样的数字**

```python
torch.zeros((2,3,4))
```

输出：

tensor([[[0., 0., 0., 0.],

​         [0., 0., 0., 0.],

​         [0., 0., 0., 0.]],



​        [[0., 0., 0., 0.],

​         [0., 0., 0., 0.],

​         [0., 0., 0., 0.]]])

```python
torch.ones((2,3,4))
```

输出：

tensor([[[1., 1., 1., 1.],

​         [1., 1., 1., 1.],

​         [1., 1., 1., 1.]],



​        [[1., 1., 1., 1.],

​         [1., 1., 1., 1.],

​         [1., 1., 1., 1.]]])

```python
torch.tensor([[1,2,3,4],[3,4,1,2],[4,3,2,1]])
```

- 通过提供包含数值的python列表（或嵌套列表）来为所需张量中的每个元素赋予确定的值

输出：

tensor([[1, 2, 3, 4],

​        [3, 4, 1, 2],

​        [4, 3, 2, 1]])

**6、常见的标准算术运算符**

```python
x = torch.tensor([1,2,3,4])
y = torch.tensor([4,3,2,1])
x + y , x - y , x * y , x / y , x ** y
```

- +
- \-
- *
- /
- ** ： 幂运算

输出：

(tensor([5, 5, 5, 5]),

 tensor([-3, -1,  1,  3]),

 tensor([4, 6, 6, 4]),

 tensor([0.2500, 0.6667, 1.5000, 4.0000]),

 tensor([1, 8, 9, 4]))

- 所有的运算都是按元素进行的

**7、指数运算**

```python
torch.exp(x)
```

输出：

tensor([ 2.7183,  7.3891, 20.0855, 54.5981])

- 也是按照元素进行的

**8、将多个张量连结在一起**

```python
x = torch.arange(12,dtype = torch.float32).reshape((3,4))
y = torch.tensor([[1,2,3,4],[3,4,1,2],[4,3,2,1]])
torch.cat((x,y),dim=0),torch.cat((x,y),dim=1)
```

- dtype = torch.float32：声明生成的张量元素的数据类型
- dim = 0：按第 0 维进行合并，即按行进行合并
- dim = 1：按第 1 维进行合并，即按列进行合并
- 这里的合并可以理解为在某个维度上进行堆叠，该维度的元素数量相加得到合并后的某一维度的元素数量

输出：

(tensor([[ 0.,  1.,  2.,  3.],

​         [ 4.,  5.,  6.,  7.],

​         [ 8.,  9., 10., 11.],

​         [ 1.,  2.,  3.,  4.],

​         [ 3.,  4.,  1.,  2.],

​         [ 4.,  3.,  2.,  1.]]),

 tensor([[ 0.,  1.,  2.,  3.,  1.,  2.,  3.,  4.],

​         [ 4.,  5.,  6.,  7.,  3.,  4.,  1.,  2.],

​         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]]))

**9、通过逻辑运算符构建二元张量**

```python
x == y
```

输出：

tensor([[False, False, False, False],

​        [False, False, False, False],

​        [False, False, False, False]])

- 按元素值进行判定

**10、对张量中的所有元素进行求和会产生一个只有一个元素的张量**

```python
x.sum()
```

输出：

tensor(66.)

- 最终得到只有一个元素的标量

1**1、即使形状不同，依然可以通过调用广播机制(bradcasting mechanism)来执行按元素操作**

```python
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
a,b
```

输出：

(tensor([[0],

​         [1],

​         [2]]),

 tensor([[0, 1]]))

```python
a + b
```

输出：

tensor([[0, 1],

​        [1, 2],

​        [2, 3]])

- a , b会自动补全称为 3 * 2 的矩阵

**12、可以使用[-1]选择最后一个元素，可以使用[1:3]选择第二和第三个元素**

```python
x,x[-1],x[1:3]
```

输出：

(tensor([[ 0.,  1.,  2.,  3.],

​         [ 4.,  5.,  6.,  7.],

​         [ 8.,  9., 10., 11.]]),

 tensor([ 8.,  9., 10., 11.]),

 tensor([[ 4.,  5.,  6.,  7.],

​         [ 8.,  9., 10., 11.]]))

- x[-1]：表示取出最后一行所有的元素
- 注意这里的x[1:3]取的是第二行和第三行的数据（这里也是左闭右开，只取了1、2，3没有取到）

**13、除了读取之外，还可以通过指定索引来将元素写入矩阵**

```python
x[1,2] = 9
x
```

输出：

tensor([[ 0.,  1.,  2.,  3.],

​        [ 4.,  5.,  9.,  7.],

​        [ 8.,  9., 10., 11.]])

**14、可以为多个元素赋相同的值，只需要索引所有的元素，然后为他们赋值**

```python
x[0:2,:] = 44
x
```

输出：

tensor([[44., 44., 44., 44.],

​        [44., 44., 44., 44.],

​        [ 8.,  9., 10., 11.]])

**15、运行一些操作可能会导致为新结果分配内存**

```python
before = id(y)
y = y + x
id(y) == before
```

- id()：返回变量在python中的唯一标识号（十进制）

输出：

False

- y不等于before，说明y之前的内存已经被析构掉了，新的y的id并不等于之前的y的id

执行原地操作

```python
z = torch.zeros_like(y)
print('id(z):',id(z))
z[:] = x + y
print('id(z):',id(z))
```

输出：

id(z): 2370001081104

id(z): 2370001081104

- z的id和之前的id并没有发生变化，并没有为z创建新的内存，这里可以理解为对z中的元素进行改写，而不是运算，也就是说，预算会改变变量的内存地址，但改写变量中的元素并不会改变元素的内存地址

**16、如果在后续计算中并没有重复使用x，也可以使用x[:] = x + y 或 x += y 来减少操作的内存开销**

```python
before = id(x)
x += y
id(x) == before
```

输出：

True

- 使用“+=”运算符并没有改变变量的内存地址
- 需要注意的是，在占用内存比较大的情况下不要进行过度复制
- 布尔值的首字母大写

**17、转换为NumPy张量**

- numpy是python中最基础的多元数组运算框架

```python
a = x.numpy()
b = torch.tensor(a)
type(a),type(b)
```

- x.numpy()：得到一个numpy的多元数组

输出：

(numpy.ndarray, torch.Tensor)

- a的type是numpy.ndarray,b的type是torch.Tensor

**18、将大小为1的张量转换为Python标量**

```python
a = torch.tensor([3.5])
a,a.item(),float(a),int(a)
```

- a.item()：返回一个numpy的**浮点数**

输出：

(tensor([3.5000]), 3.5, 3.5, 3)



**数据预处理**

- 对于原始数据，如何进行读取，使得能够使用机器学习的方法进行处理

**1、创建一个人工数据集，并存储在csv（逗号分隔值）文件**

```python
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')
with open(data_file,'w') as f:
  f.write('NumRooms,Alley,Price\n') # 列名
  f.write('NA,Pave,127500\n') # 每行表示一个数据样本
  f.write('2,NA,10600\n')
  f.write('4,NA,178100\n')
  f.write('NA,NA,140000\n')
```

- 创建一个文件，文件名叫做house_tiny.csv
- csv：每一行是一个数据，每个域（entry）用逗号分开
- NA：not a number，未知数

**2、从创建的csv问及那中加载原始数据集**

- 一般使用pandas库进行csv文件的读取

```python
impot pandas as pd
data = pd.read_csv(data_file)
print(data)
```

输出：

  NumRooms  Alley     Price

0      NaN   Pave  127500.0

1     2，NA  10600       NaN

2        4    NaN  178100.0

3      NaN    NaN  140000.0

注意：

- 如果报错”no module named pandas“，在终端使用”pip install pandas“命令即可安装pandas
- 如果程序报错”UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 39: invalid start byte“，需要将1中的with open(data_file,'w') as f:这行代码改成with open(data_file,'w',encoding='utf-8') as f:即可
- 如果不使用print函数输出data，则会以html的格式输出data，如下图所示：

![img](https://i0.hdslb.com/bfs/note/1236ed7423afa298ee64923163ff5a718170b8d6.png)



**3、为了处理缺失的数据，典型的方法包括插值和删除，此处考虑使用插值的方法**

```python
inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

输出：

   NumRooms Alley

0       3.0  Pave

1       2.0   NaN

2       4.0   NaN

3       3.0   NaN

- 将NA使用该列其它非NA值的平均值进行填充

**4、对于inputs中的类别值或者离散值，将”NaN“视为一个类别**

```python
inputs = pd.get_dummies(inputs,dummy_na=True)
print(inputs)
```

- dummy_na=True：将Na也变成一个类，并赋值为1

输出：

   NumRooms  Alley_Pave  Alley_nan

0       3.0           1          0

1       2.0           0          1

2       4.0           0          1

3       3.0           0          1



对于不是数值的数据的处理方法

将缺失值做成一个特别的类，然后将其变成一个数值（而不是字符串，字符串相对来讲不好处理），对数值的处理：将所有出现的不同种的值都变成一个特征

**5、现在已经把所有缺失的值和字符串变成了数值，inputs和outputs中的所有的条目都是数值类型，他们可以转换为张量格式**

```python
import torch
x , y = torch.tensor(inputs.values),torch.tensor(outputs.values)
x , y
```

输出：

(tensor([[3., 1., 0.],

​         [2., 0., 1.],

​         [4., 0., 1.],

​         [3., 0., 1.]], dtype=torch.float64),

 tensor([127500,  10600, 178100, 140000]))

- dtype=torch.float64：传统的python会默认使用64位的浮点数，但是64位的浮点数一般计算比较慢，对于深度学习来讲，通常使用32位浮点数


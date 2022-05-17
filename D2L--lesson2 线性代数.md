# D2L--lesson2

## **线性代数**

![img](https://i0.hdslb.com/bfs/note/795ac9e48a6f4fe17d6fd8b3299183d281216a1a.png)

### **多维数组和线性代数的区别？**

- 多维数组是一个计算机的概念，纯计算机的语言
- 线性代数是从数学上的表达，具有数学上的意义

### **标量**

- 可以理解为一个单个的值

**简单操作**

- c = a + b
- c = a * b
- c = sin(a)

**长度**

![img](https://i0.hdslb.com/bfs/note/2e6798a1a1e650e5a807a0985bd7d34b0d2d310e.png)



### **向量**

- 可以理解为一行值，既可以是行向量，也可以是列向量

**简单操作**

![img](https://i0.hdslb.com/bfs/note/3177af94d717d78d68b7caadffde5be23f08661b.png)

**长度**

![img](https://i0.hdslb.com/bfs/note/9b21019eb3d7bf683ef6e2e7d514ca27b4c4c8ab.png)

![img](https://i0.hdslb.com/bfs/note/176478b86636d22d475484ec766e1776d728870e.png)



**点乘**

![img](https://i0.hdslb.com/bfs/note/1be3876d9d06481ba9a564a48aa05718857c5ec3.png)

- 对应元素相同

**正交**

![img](https://i0.hdslb.com/bfs/note/e4708d29ca2ac1fd5840bf70cbb93a06e46d8818.png)

- 两个向量如果是垂直的话，他们的点乘的值为0

### **矩阵**

一般用大写字母表示矩阵，下面为了方便用小写字母表示

**简单操作**

![img](https://i0.hdslb.com/bfs/note/2860ef1597645afc8b6524ec64ef7cd830687518.png)

**乘法**

- 矩阵乘以向量（从直观上来讲是一个扭曲的空间）

  **线性代数 P1 - 04:43**

![img](https://i0.hdslb.com/bfs/note/567a231b77d9bbf8dacc9278b88332bfc745c66b.png)

- 矩阵乘以矩阵

![img](https://i0.hdslb.com/bfs/note/9abb30a71f34c1d12cb1b214b9668b27829dd5b0.png)

**范数（矩阵的长度）**

![img](https://i0.hdslb.com/bfs/note/11cf903d102f68022b956531adb289a79ff93317.png)

![img](https://i0.hdslb.com/bfs/note/96a27233a5eb323e6cbdb417ec4e421360ef2077.png)

![img](https://i0.hdslb.com/bfs/note/19ec2b6751810f4f501155a68205aae485e60cbb.png)

![img](https://i0.hdslb.com/bfs/note/50a0c4409c9db1cd9dfa9a1a3c13bc23885cfedc.png)

![img](https://i0.hdslb.com/bfs/note/53733cc1b9a11588d16c3f040960ebb53e938d36.png)

- 对称矩阵总是能找到特征向量，但不是每个矩阵都能找到特征向量
- 

### **线性代数实现**

**1、标量由只有一个元素的张量表示**

```python
import torch
x = torch.tensor([3.0])
y = torch.tensor([2.0])
x + y , x * y , x / y , x ** y 
```

输出：

(tensor([5.]), tensor([6.]), tensor([1.5000]), tensor([9.]))

**2、可以将向量视为标量值组成的列表**

```python
x = torch.arange(4)
x
```

输出：

tensor([0, 1, 2, 3])

**3、通过张量的索引来访问任一元素**

```python
x[3]
```

输出：

tensor(3)

**4、访问张量的长度**

```python
len(x)
```

输出：

4

**5、只有一个轴的张量，形状只有一个元素**

```python
x.shape
```

输出：

torch.Size([4])

**6、通过指定两个分量m和n来创建一个形状为 m \* n 的矩阵**

```python
a = torch.arange(20).reshape(5,4)
a
```

输出：

tensor([[ 0,  1,  2,  3],

​        [ 4,  5,  6,  7],

​        [ 8,  9, 10, 11],

​        [12, 13, 14, 15],

​        [16, 17, 18, 19]])

**7、矩阵的转置**

```python
a.T
```

输出：

tensor([[ 0,  4,  8, 12, 16],

​        [ 1,  5,  9, 13, 17],

​        [ 2,  6, 10, 14, 18],

​        [ 3,  7, 11, 15, 19]])

**8、对称矩阵(symmetric matrix)**

**对称矩阵等于其转置：a = aT**

```python
b = torch.tensor([[1,2,3],[2,3,1],[3,2,1]])
b
```

输出：

```python
tensor([[1, 2, 3],
       [2, 3, 1],
       [3, 2, 1]])
b == b.T
```

输出：

tensor([[ True,  True,  True],

​        [ True,  True, False],

​        [ True, False,  True]])

- 矩阵的所有元素都是关于对角线对称的

**9、就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构**

```python
x = torch.arange(24).reshape(2,3,4)
x
```

输出：

tensor([[[ 0,  1,  2,  3],

​         [ 4,  5,  6,  7],

​         [ 8,  9, 10, 11]],



​        [[12, 13, 14, 15],

​         [16, 17, 18, 19],

​         [20, 21, 22, 23]]])

- tensor中的行是最后一维，列是倒数第二维，以此类推

**10、给定具有相同形状的任何两个张量，任何按元素二元计算的结果都将是相同形状的张量**

```python
a = torch.arange(20,dtype = torch.float32).reshape(5,4)
b = a.clone()
a , a + b
```

- 通过分配新内存，将 a 的一个副本分配给 b

输出：

(tensor([[ 0.,  1.,  2.,  3.],

​         [ 4.,  5.,  6.,  7.],

​         [ 8.,  9., 10., 11.],

​         [12., 13., 14., 15.],

​         [16., 17., 18., 19.]]),

 tensor([[ 0.,  2.,  4.,  6.],

​         [ 8., 10., 12., 14.],

​         [16., 18., 20., 22.],

​         [24., 26., 28., 30.],

​         [32., 34., 36., 38.]]))

11、两个矩阵的按元素乘法称为 哈达玛积(Hadamard product) 数学符号是⭕里面一个点

![img](https://i0.hdslb.com/bfs/note/33cfe0d4230d08cc2a3a4999642b89db41f07cbb.png)

```python
a * b
```

输出：

tensor([[  0.,   1.,   4.,   9.],

​        [ 16.,  25.,  36.,  49.],

​        [ 64.,  81., 100., 121.],

​        [144., 169., 196., 225.],

​        [256., 289., 324., 361.]])

```python
a = 2
x = torch.arange(24).reshape(2,3,4)
a + x , ( a * x ).shape
```

输出：

(tensor([[[ 2,  3,  4,  5],

​          [ 6,  7,  8,  9],

​          [10, 11, 12, 13]],

 

​         [[14, 15, 16, 17],

​          [18, 19, 20, 21],

​          [22, 23, 24, 25]]]),

```python
 torch.Size([2, 3, 4]))
```

- 张量和一个标量进行运算实际上就是张量的所有元素和这个标量进行运算

**12、计算其元素的和**

```python
x = torch.arange(4,dtype=torch.float32)
x , x.sum()
```

输出：

(tensor([0., 1., 2., 3.]), tensor(6.))

**13、表示任意形状张量的元素和**

```python
a = torch.arange(20,dtype = torch.float32).reshape(5,4)
a.shape , a.sum()
```

输出：

(torch.Size([5, 4]), tensor(190.))

- .sum()：不管张量是什么形状，计算出来的结果始终是一个标量

```python
a = torch.arange(20 * 2,dtype = torch.float32).reshape(2,5,4)
a.shape , a.sum()
```

输出：

(torch.Size([2, 5, 4]), tensor(780.))

**14、指定求和张量的值**

```python
a_sum_axis0 = a.sum(axis = 0)
a_sum_axis0 , a_sum_axis0.shape
```

输出：

(tensor([[20., 22., 24., 26.],

​         [28., 30., 32., 34.],

​         [36., 38., 40., 42.],

​         [44., 46., 48., 50.],

​         [52., 54., 56., 58.]]),

```python
 torch.Size([5, 4])
a_sum_axis1 = a.sum(axis = 1)
a_sum_axis1 , a_sum_axis1.shape
```

输出：

```python
(tensor([[ 40.,  45.,  50.,  55.],
         [140., 145., 150., 155.]]),
 torch.Size([2, 4]))
a_sum_axis2 = a.sum(axis = 2)
a_sum_axis2 , a_sum_axis2.shape
```

输出：

(tensor([[  6.,  22.,  38.,  54.,  70.],

​         [ 86., 102., 118., 134., 150.]]),

 torch.Size([2, 5]))

```python
a.sum(axis = [0 , 1]).shape
```

输出：

torch.Size([4])

**15、一个与求和相关的量是 平均值(mean或average)**

```python
a = torch.arange(20,dtype = torch.float32).reshape(5,4)
a.mean() , a.sum() / a.numel()
```

- 等价于对元素求和然后除以元素的个数

输出：

(tensor(9.5000), tensor(9.5000))

```python
a.mean(axis = 0) , a.sum(axis = 0) / a.shape[0]
```

- 等价于求和然后除以维度的形状

输出：

(tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))

**16、计算综合或均值时保持轴数不变**

```python
sum_a = a.sum(axis = 1 , keepdims = True)
sum_a
```

- keepdims=True：使被求和的维度大小变为1，为1的好处是可以利用广播机制

输出：

tensor([[ 6.],

​        [22.],

​        [38.],

​        [54.],

​        [70.]])

**17、通过广播将 a 除以sum_a**

```python
a / sum_a
```

输出：

tensor([[0.0000, 0.1667, 0.3333, 0.5000],

​        [0.1818, 0.2273, 0.2727, 0.3182],

​        [0.2105, 0.2368, 0.2632, 0.2895],

​        [0.2222, 0.2407, 0.2593, 0.2778],

​        [0.2286, 0.2429, 0.2571, 0.2714]])

**18、某个轴计算a元素的累积总和**

```python
a.cumsum(axis = 0)
```

输出：

tensor([[ 0.,  1.,  2.,  3.],

​        [ 4.,  6.,  8., 10.],

​        [12., 15., 18., 21.],

​        [24., 28., 32., 36.],

​        [40., 45., 50., 55.]])

**19、点积是相同位置的按元素乘积的和**

```python
y = torch.ones(4,dtype=torch.float32)
x , y , torch.dot(x,y)
```

输出：

(tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))

**20、可以通过执行按元素乘法，然后进行求和来表示两个向量的点积**

```python
torch.sum(x * y)
```

输出：

tensor(6.)

**21、**

![img](https://i0.hdslb.com/bfs/note/819e866460c13fdff28ae39d3aace2c547ce296d.png)

```python
a.shape , x.shape , torch.mv(a,x)
```

- torch.mv()：做矩阵向量的乘积

输出：

(torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))

**22、可以将矩阵-矩阵乘法AB看作简单地执行m次矩阵-向量积，并将结果拼接到一起，形成一个n \* m矩阵**

```python
b = torch.ones(4,3)
torch.mm(a,b)
```

输出：

tensor([[ 6.,  6.,  6.],

​        [22., 22., 22.],

​        [38., 38., 38.],

​        [54., 54., 54.],

​        [70., 70., 70.]])

**23、L2范数是向量元素平方和的平方根**

![img](https://i0.hdslb.com/bfs/note/859e4a5e9d6098c03d7b2e4f46b6829ecbc0dc04.png)

```python
u = torch.tensor([3.0,-4.0])
torch.norm(u)
```

输出：

tensor(5.)

**24、L1范数表示为向量元素的绝对值之和**

![img](https://i0.hdslb.com/bfs/note/03b0fc72cdad05fc089c132cd83144a77d8c6efe.png)

```python
torch.abs(u).sum()
```

输出：

tensor(7.)

**25、矩阵的 弗罗贝尼乌斯范数(Frobenius norm) 实矩阵元素的平方和的平方根**

![img](https://i0.hdslb.com/bfs/note/bed8b52cd2aabab7b57af287cfe2ac7ee78686c4.png)

```python
torch.norm(torch.ones((4,9)))
```

输出：

tensor(6.)

- 等价于将矩阵拉成一个向量，然后做一个向量的范数
- 计算简单，是最常用的范数

**按待定轴求和**

**按特定轴求和 P3 - 00:03**

总的来说axis等于几就去掉那一维（keepdim不为True时），当keepdim为True时,将axis指定的那一维的shape置为1

```python
import torch
a = torch.ones(2,5,4)
a.shape
```

输出：

torch.Size([2, 5, 4])

```python
a.sum().shape
```

输出：

torch.Size([])

- shape为空，表示它是一个标量

```python
a.sum(axis = 1).shape
```

输出：

torch.Size([2, 4])

```python
a.sum(axis = 1)
```

输出：

tensor([[5., 5., 5., 5.],

​        [5., 5., 5., 5.]])

```python
a.sum(axis = 0).shape
```

输出：

torch.Size([5, 4])

```python
a.sum(axis = [0,2]).shape
```

输出：

torch.Size([5])

```python
a.sum(axis = 1,keepdim=True).shape
```

输出：

torch.Size([2, 1, 4])

- 维度的个数没有发生变化只是第2个元素（下标为1）变换成了1
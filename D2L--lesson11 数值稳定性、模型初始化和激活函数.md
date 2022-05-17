# D2L--lesson11数值稳定性、模型初始化和激活函数

## **数值稳定性**

### **神经网络的梯度**

![img](https://i0.hdslb.com/bfs/note/e9e3b53b3bdc6897762b1d8b1bff9874d6faeef3.png)

- t表示第t层
- h（t-1）表示第t-1层的输出
- 所有的h都是向量，向量关于向量的导数是一个矩阵
- 太多的矩阵乘法会导致：

![img](https://i0.hdslb.com/bfs/note/418570915f283db783558f2e426145144c436418.png)

**举例：MLP**

![img](https://i0.hdslb.com/bfs/note/eb3a4fafd676a7578b9e307f97d997dc32063639.png)

- **梯度爆炸****：**

![img](https://i0.hdslb.com/bfs/note/a12957ce333c2394e9de473fb3f48d84c85d52cb.png)

- 梯度爆炸所带来的问题

![img](https://i0.hdslb.com/bfs/note/e681b436387074349b4c4a71f6148a8c98c7526f.png)

- 16位浮点数最大的问题是它的数值区间比较小
- 不是不能训练，只是对于学习率的调整比较难调
- **梯度消失****：**

![img](https://i0.hdslb.com/bfs/note/5d1fe619e5632adc4b451c920d1c217e009203af.png)

- 

![img](https://i0.hdslb.com/bfs/note/d42b09a4967851a90c7a39a92e5cc9adcfbc8860.png)

- 梯度消失的问题

![img](https://i0.hdslb.com/bfs/note/20617eb2b3d585eac9bfdbac5f0a3b1872ba0e13.png)

- 对顶部的影响比较小，因为做矩阵乘法的次数比较少
- 仅仅顶部层的训练的较好，无法让神经网络更深，就等价于是浅层的神经网络

**总结**

![img](https://i0.hdslb.com/bfs/note/a0f41b8de66c3f3ab88583a3d6176c87ea87d5a7.png)

- 尽量避免梯度过大或者过小



### **让训练更加稳定**

![img](https://i0.hdslb.com/bfs/note/725dfee8ad5d5a9059b19ae4838bbb5b77a138da.png)



## **合理的权重初始化和激活函数**

让每一层的方差是一个常数

![img](https://i0.hdslb.com/bfs/note/8293cd1d8618dcea0baea42f11ce434254304f35.png)

- E：均值
- Var：方差

### **权重初始化**

![img](https://i0.hdslb.com/bfs/note/1bfdac53d2e374f1871ca8af560655ec110021d7.png)

- 在训练开始的时候更容易有数值不稳定

![img](https://i0.hdslb.com/bfs/note/5d94e317ef1debc3e5e9306ebc86afd98a6f6201.png)

- iid：independent identically distributed，独立同分布
- 前一层的输出（当前层的输入）和当前层的权重是相互独立的
- 当X和Y相互独立时， E(XY)=E(X)E(Y)

![img](https://i0.hdslb.com/bfs/note/116d3fbf4adb8b4a59072e6a81ee5725d2c6dbdf.png)

- 输入方差和输出方差相同，则可以推出紫色圈出的内容

![img](https://i0.hdslb.com/bfs/note/f42c49448c1306f4882dc45fa05efd44833b2ba3.png)

### **Xavier初始化**

![img](https://i0.hdslb.com/bfs/note/12bb978e9676670ddd4fe91dd179124451e60e43.png)

- 第一个条件是使得每次前向输出的方差是一致的
- 第二个条件是使得梯度是一样的
- 除非输入等于输出，否则无法同时满足这两个条件
- γt：第t层的权重的方差
- 给定当前层的输入和输出的大小，就能确定权重所要满足的方差的大小
- Xavier是常用的权重初始化方法：权重初始化的时候的方差是根据输入和输出维度来定的

**在有激活函数的情况下**

![img](https://i0.hdslb.com/bfs/note/0b1fe2fe6840b26f2089a7648520fe2692469422.png)

- 激活函数的输入和输出的方差有a^2倍的关系，激活函数如果将值放大a倍的话，它的方差会被放大a^2倍
- 如果要使激活函数不改变输入输出的方差，则a=1：为了使得前向输出的均值和方差都是均值为0，方差为固定的话，激活函数只能是β=0，a=1，即激活函数必须是等于本身

![img](https://i0.hdslb.com/bfs/note/796742c00800672c74956ee501efbc91a96b2024.png)

- 

![img](https://i0.hdslb.com/bfs/note/19a8047bda033d65eebc4c41b305b84814272ae0.png)

- 在0点附近的时候tanh和relu基本满足f(x)=x的关系，sigmoid需要做出调整
- 图中蓝色曲线是经过调整之后的sigmoid激活函数，调整之后就能够解决sigmoid激活函数本身所存在的一些问题

**总结**

![img](https://i0.hdslb.com/bfs/note/d5e3cef20f32f435fb36606501bfde2ab1cabf60.png)

- 使得每一层的输出和每一层的梯度都是均值为0，方差为固定数的随机变量
- 权重初始化可以使用Xavier，激活函数可以使用tanh和relu，如果选择sigmoid的话需要进行调整
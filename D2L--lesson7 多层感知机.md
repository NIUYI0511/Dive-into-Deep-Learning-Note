# D2L--lesson7

## **感知机**

### 感知机

![img](https://i0.hdslb.com/bfs/note/0a0ffef2ea1ab9ad01ca081725550aa25b95ae19.png)

- 人工智能最早的模型
- x ，w 都是向量，b 是一个标量
- <w , x>：w 和 x 做内积
- 感知机其实就是一个二分类问题：输入大于0就输出1，否则输出0

![img](https://i0.hdslb.com/bfs/note/d5d49c78daeebedf1207f4e76f7fa91cd4ccc68e.png)

- 和线性回归的不同点在于线性回归输出的是一个实数而感知机输出的是一个离散的类
- 和softmax的区别是，在有n个类的情况下，他会输出n个元素，所以可以是一个多分类的问题，而这里只输出一个元素，最多只能做一个二分类问题

### **训练感知机**

![img](https://i0.hdslb.com/bfs/note/1e81290de8d654150e16435a201e453512b12751.png)

- 预测值和实际值不符的话（异号）会导致他们的乘积小于等于零，从而更新权重

### **收敛定理**

![img](https://i0.hdslb.com/bfs/note/770e19412b5ba5fb17d0a489a8beb83ae1fd6826.png)

- 收敛定理确定停止的条件
- p大于等于0

### **感知机不能拟合异或函数**

![img](https://i0.hdslb.com/bfs/note/46b99eeb9b28cf305f1336d098c9b77cf5a8d7d8.png)

- 无法使用一条直线将图上的四个点分成两类

**总结**

![img](https://i0.hdslb.com/bfs/note/ba8833d635e7fba1c3547f7241d538debaa27588.png)



### **多层感知机**

![img](https://i0.hdslb.com/bfs/note/8571be8d4d0d48943f3f39cbfdbaba45bae3cc68.png)

**异或问题**

![img](https://i0.hdslb.com/bfs/note/1672e35ee589ff53e3275a1e71efe0f30e6a269e.png)

- 组合两个函数，一层变成了多层

**单隐藏层**

![img](https://i0.hdslb.com/bfs/note/49262f35ce4a088222acda3140be48e133e8dac1.png)

- 输入层的大小是固定的，输出层的大小等于类别的数量，唯一可以设置的是隐藏层的大小

**单分类问题**

![img](https://i0.hdslb.com/bfs/note/effe47b914e7386101b185da91764768d6d774c8.png)

- 为什么需要非线性激活函数？线性的激活函数或导致最终输出还是一个线性函数，就等价于一个单层的感知机了

### **激活函数**

sigmoid激活函数

![img](https://i0.hdslb.com/bfs/note/426f357a704a4bba03cb6574fee91a4b0e51bae4.png)

- 将 x 的值投影到一个0和1的开区间中
- sigmoid实际上是阶跃函数的温和版

tanh激活函数

![img](https://i0.hdslb.com/bfs/note/3a0cf70c1fa18150a1292b383a293a459ab2dd6c.png)

- 和sigmoid很像，区别在于它是将输入投影到-1到1的区间内
- -2是为了方便求导

**ReLU激活函数**

![img](https://i0.hdslb.com/bfs/note/8db69dd02407c4fdf7a4bb73bc6f48c3921afe8b.png)

- 最常用
- 不用做指数运算

### **多类分类**

![img](https://i0.hdslb.com/bfs/note/d671c0fd33dd6c1b00ca194b2e3c0a7ee33f4017.png)

- softmax就是将所有的输入映射到0和1的区间之内，并且所有输出的值加起来等于1，从而转变成概率

![img](https://i0.hdslb.com/bfs/note/57e5313981071a91af4858cae64691ab70ad5aef.png)

- 和单分类的区别在于最后的输出做了一个softmax操作

### **多隐藏层**

![img](https://i0.hdslb.com/bfs/note/5bbc7edc81e28b8a8a59d52140300b3e2f7a0ff8.png)

- 超参数变多了

**总结**

![img](https://i0.hdslb.com/bfs/note/efcbf536962602046b00e213ec48c971b2a5c389.png)
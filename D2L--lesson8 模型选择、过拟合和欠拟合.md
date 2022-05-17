# D2L-lesson8

## **模型选择**

![img](https://i0.hdslb.com/bfs/note/5dbb67efd35eadf6a846ec03056c16e0ab9d3b41.png)

- 我们其实更关注泛化误差
- 训练误差是模型在有标签的数据上的误差
- 泛化误差是在新的数据上的误差

### **如何计算训练误差和泛化误差？**

一般有两种数据集：

- 验证数据集：评估模型好坏的数据集（多层感知机的模型大小、学习率大小等是通过验证数据集来进行确定的）
- 测试数据集

![img](https://i0.hdslb.com/bfs/note/356f753d972e28ad9096f300a5482285b0f5805b.png)

- 通常将训练数据集对半分，一半的数据集用来训练模型，另一半（验证数据集）用来测试精度，根据精度的大小来对模型的超参数进行调整，然后再在验证数据集上测试精度，重复操作来调整超参数
- 验证数据集没有参与训练，所以在一定程度上确实能够反映超参数选择的好坏
- 验证数据集一定不能与训练数据集混在一起（ImageNet数据集和谷歌搜出来的图片是有交集的，不能用测试数据集的结果作为调整超参数的依据，相当于考试作弊）
- 测试数据集只能被使用一次
- 验证数据集上的精度也可能是虚高的，超参数很有可能是在验证数据集上调出来的，所以可能导致验证数据集上的精度可能不能代表在新数据上的泛化能力

### K折交叉验证

**经常会遇到没有那么多训练数据的问题，没有足够多的数据来进行使用**，解决这个问题常见的方法是K-fold cross-validation(K-折交叉验证)

![img](https://i0.hdslb.com/bfs/note/857f7bb1af551b0f0fe464c5b096f0b01124d1f2.png)

- 拿到数据集后先进行随机打乱，然后再分割成K块，每一次将第K块数据集作为验证数据集，其余的作为训练数据集

![img](https://i0.hdslb.com/bfs/note/ae1931d63f83043dfee21479b2a4083e52f2db48.png)

- 数据集大的话K可以取5或者更小，数据集大的话K可以取10或者更大

**总结**

![img](https://i0.hdslb.com/bfs/note/c73e78673baaf70a399c3c4c87abf1b92ca75d07.png)



### **过拟合和欠拟合**

![img](https://i0.hdslb.com/bfs/note/12dfd6c280811ffcfbc52120e148163b04ddacaf.png)



![img](https://i0.hdslb.com/bfs/note/e61775f078520f1bfe08104f9e15ba70a2cc0aed.png)

- 根据数据集的复杂度来选择对应的模型容量
- 过拟合可能导致模型的泛化能力差
- 欠拟合可能导致模型精度低

### **模型容量**

![img](https://i0.hdslb.com/bfs/note/36a32b89dc0b7fa924322bb6fa195ad97d489957.png)



![img](https://i0.hdslb.com/bfs/note/752d531e4bdaf6e58d8f5bee95e991d7728bc7aa.png)



**模型容量的影响**

![img](https://i0.hdslb.com/bfs/note/a79b8dd3a59c7404f34831423806489677edb1c9.png)

- 训练误差并不是越高越好，如果数据中含有大量的噪音可能导致模型训练不起作用，并且模型的泛化能力变差
- 真正关心的是泛化误差

![img](https://i0.hdslb.com/bfs/note/7e0dc27ae7ac6a0e684ee37dbb2ae95b99090230.png)

- 核心任务是要使泛化误差的最优点降低，并且缩小在该最优点同一模型容量下泛化误差与训练误差之间的差距（有时候为了降低泛化误差可能会选择接受一定程度的过拟合）
- 过拟合本身不是一件坏事，首先是模型的容量要充足，然后再去控制模型的容量，使得泛化误差下降（深度学习的核心）

**估计模型容量**

![img](https://i0.hdslb.com/bfs/note/94bcd035464f089bb773932b995cd58de487c2e8.png)

- 参数值的选择范围越大，模型的复杂度就越高

### **统计理论的核心思想**：VC维

![img](https://i0.hdslb.com/bfs/note/299f598b87126e8f214ae0f086dfcfc61a93c3cf.png)



![img](https://i0.hdslb.com/bfs/note/126c6724fc7cac6aaa97b9ba96fcd94a9f8ec284.png)



![img](https://i0.hdslb.com/bfs/note/cc0b162ea61fb643100cdd365399116c89662976.png)

### **数据复杂度**



![img](https://i0.hdslb.com/bfs/note/07ad5888c4097a8cdfc1864fbb8e11bcb3b9318d.png)

- 多样性是指数据集中类别的多少

**总结**



![img](https://i0.hdslb.com/bfs/note/44a56420f0d4439353255ccc3a0411ba1ad84009.png)
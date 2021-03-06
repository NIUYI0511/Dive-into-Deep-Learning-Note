# D2L--lesson9

## **权重衰退**

### **最常见的处理过拟合的方法**

如何控制模型的容量

- 将模型变得比较小，减少里面的参数的数量
- 缩小参数值的取值范围

### **硬性限制**

![img](https://i0.hdslb.com/bfs/note/c8903213a67b827d9c5b72494d00da6eb4e81cc1.png)

- θ用来限制权重 w 的变化范围
- 通常不会限制偏移 b ,从统计学上来讲，偏移是整个数据对于 0 点的偏移，是不应该限制的，但是实际上，限不限制效果相同
- θ越小，限制就越强。最强的情况下就是θ等于0，所有的w都等于0，只能选一个偏移
- 一般来说θ会选择1、0.1、0.01

### **柔性限制**

![img](https://i0.hdslb.com/bfs/note/3deffd1c549a308f66b68df1e75b6f10cd2a5263.png)

- λ是一个超参数，λ控制了整个正则项的重要程度
- λ趋向于无穷大的时候就等价于硬性限制中θ趋向于0，使得最优解w*也会慢慢趋向于0
- 可以通过增加λ来控制模型的复杂度（让模型不要太复杂）

### **演示对最优解的影响**

![img](https://i0.hdslb.com/bfs/note/8fe33336fbba4160773c7911cd6c2bfea24083ef.png)

- 绿线代表损失函数l的等高线
- 绿点代表损失函数l的最优点（只优化损失的情况）
- w的2次项可以认为是一个以原点为中心的等高线，如橘黄色圆圈
- 原始的最优解（绿点）就不是最优了，因为它的值对于橘黄色的线来说比较大。这里可以理解为w~*就是橘黄色圆圈和绿色圆圈的等高线的值之和，在橘黄色圆圈中，原点值最小，向外增加；在绿色圆圈中，绿点值最小向外增加
- 如果w~*从绿点出发，沿着蓝色箭头走，l的值会增大，但是w的二次项（阀的项）的值会减小，走到w*处达到平衡点总体上来讲，阀的引入，使得最优解向原点偏移，对应的最优解的值会变得小一些，绝对值会变小，从而模型的复杂度会变低

### **参数更新法则**

![img](https://i0.hdslb.com/bfs/note/af7f41eeca4a0e594315281b7d26681dca73d021.png)

- 通常来讲，λ和学习率的乘积是小于1的
- 为什么叫权重衰退？因为λ的引入使得当前的权重做了一次缩小操作，即所说的衰退

**总结**



![img](https://i0.hdslb.com/bfs/note/e403aa6ed1c817371c222b0b2dca4334ea10c709.png)

- λ是控制模型的超参数，通过控制λ的大小来控制模型的复杂度
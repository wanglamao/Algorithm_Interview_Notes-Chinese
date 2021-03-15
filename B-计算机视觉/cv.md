# Loss

## GIoU(Generalized IoU)

定义：

1.$IoU=\frac{A\cap B}{A\cup B}$

2.$GIoU=IoU -\frac{|c\backslash (A\cap B)|}{|C|}$

3.$L_{IoU}=1-IoU$

4,$L_{GIoU}=1-GIoU$

#### 优点/改进：

反映了两个框的距离

#### ref

https://blog.csdn.net/weixin_41735859/article/details/89288493

## Smooth L1 Loss

````math

Smooth L_1=\begin{cases}
0.5x^2,|x|<1,\\
|x|-0.5,otherwise
\end{cases}
````

#### 优点：回避了L1和L2的缺点

从损失函数对x的导数可知： ![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D)损失函数对x的导数为常数，在训练后期，x很小时，如果learning rate 不变，损失函数会在稳定值附近波动，很难收敛到更高的精度。 ![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D) 损失函数对x的导数在x值很大时，其导数也非常大，在训练初期不稳定。![[公式]](https://www.zhihu.com/equation?tex=smooth_%7BL_%7B1%7D%7D%5Cleft%28+x+%5Cright%29+)** 完美的避开了 **![[公式]](https://www.zhihu.com/equation?tex=L_%7B1%7D)**和 **![[公式]](https://www.zhihu.com/equation?tex=L_%7B2%7D)**损失的缺点。**

#### 缺点：与IoU不等价

#### ref

https://zhuanlan.zhihu.com/p/104236411

## DIoU(Distance-IoU Loss)

考虑了**重叠面积**和**中心点距离**

IoU-based loss:

$L=1-IoU+R(B,B^{gt})$，其中$R(B,B^{gt})$代表惩罚项

DIoU：

$L_{DIoU}=1-IoU+\frac{\rho^2(b,b^{gt})}{c^2}$，其中$\rho$为欧氏距离，$c$为C（最小包括框）的对角线长度

#### 优点

比GIoU收敛快

## CIoU(Complete IoU)

$R_{CIoU}=\frac{\rho^2(b,b^{gt})}{c^2}+\alpha\nu$

#### ref

https://zhuanlan.zhihu.com/p/104236411

## Focal Loss

## Cross Entropy Loss

* Self-information:$I(X)=-log(P(X))$$

* Entropy: Expectation of self-information:$H(X)=E_{X \sim P}[I(X)]$

* KL Divergence: $D_{KL}(P||Q)=\sum_{x\in X}P(x)(logP(x)-log(Q(X))$

````
expectation of the log difference between the probability of data in the original distribution with the approximating distribution. 

P=>real distribution Q=>our distribution
````

* Cross-Entropy: $H_P(Q)=-E_{X\sim P}log(Q(X))=-\sum_{x\in X}P(x)log(Q(x))$

#### relationship

$H_P(Q)-H(X)=D_P(Q)$



# Activation Function

## ReLU

解决梯度消失：相对于sigmoid和tanh，|输入值|较大时，导数很小

dying ReLU: 

假设一个神经元(neuron)是Relu(wx+b)Relu(wx+b)。因为一般用mini batch（SGD）优化算法，每次计算gradient只用一组（batch）数据点。假如用一组数据点更新w,bw,b后，其余数据点wx+b<0wx+b<0，那么只会有一组点能通过这个neuron并更新它的参数，对于绝大多数点来说，不能通过这个neuron，也不能更新参数，相当于“死掉”。如果dying relu 很多，对于大多数数据来说神经网络大部分通路断掉，学习能力就变弱了。

解决办法是用leakyRelu等；bias的初始值设为正数，比如1；减小learning rate。

## Leaky ReLU

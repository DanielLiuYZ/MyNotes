# Machine Learning

Author: Daniel Liu
Contact me:  191240030@smail.nju.edu.cn
Course: Machine Learning - AndrewNg

[TOC]

## L1 - 引言 Introduction 

### 1 机器学习

**机器学习定义(Tom Mitchell, 1998)：**A computer program is said to learn from **experience E** with respect to some **task T** and some **performance measure P**, if its performance on T, as measured by P, improves with experience E. 

**机器学习算法：**

+ 监督学习 supervised learning
+ 无监督学习 unsupervised learning
+ 其他：强化学习 reinforcement learning，推荐系统 recommender systems

### 2 监督学习

**监督学习：**在训练数据中给出“正确答案”

**两类问题：**

+ 回归问题 regression：预测连续值输出
+ 分类问题 classification：预测离散值输出

通常模型中有多个属性需要考虑，一种比较重要的情况是存在无数种属性

### 3 无监督学习

**无监督学习：**训练数据中没有标签/标签相同，需要在数据集中找到某种结构

**常见算法：**

+ 聚类算法 clustering algorithm：将数据分为不同的簇
+ 鸡尾酒会算法 cocktail party algorithm：将叠加的数据集分离

***

## L2 - 单变量线性回归 Univariate Linear Regression

### 1 单变量线性回归

监督学习中，在训练集上用学习算法进行训练，输出函数*h*.
单变量线性回归中，将*h*表示为$h_{\theta}(x)=\theta_0+\theta_1x$.

Notation:
m: 训练集样本数目
$(x^{(i)},y^{(i)})$: 训练数据
$\theta_i$: 模型参数

### 2 代价函数

Hypothesis: $h_{\theta}(x)=\theta_0+\theta_1x$。我们需要选取的$\theta_0, \theta_1$来尽可能准确地拟合训练集，这里就引入了**代价函数**，最常用的代价函数是**平方误差代价函数**：
$$
J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2
$$
目标是最小化$J(\theta_0, \theta_1)$

### 3 梯度下降算法

通过梯度下降法 Gradient Descendent，可以找到函数$J(\theta_0, \theta_1,\cdots,\theta_n)$的局部最小值。其核心思路是先选取一个初始点$(\theta_0, \theta_1)$(注意初始点的选取可能影响得到的局部最小值)，然后不断改变$\theta_0, \theta_1$直到得到最小值。

**梯度下降算法：**
repeat until convergence {
	$\theta_j := \theta_j-\alpha\frac{\part}{\part \theta_j}J(\theta_0, \theta_1,\cdots,\theta_n)$, $j=1,2,\cdots,n$
}

$\theta_i$要同步更新，因而在实现时要先用temp变量先保存结果。
$\alpha$是学习速率，取值太小会导致梯度下降过程太慢，太大会导致无法收敛乃至发散。$\alpha$取常数即可，因为偏导数会在接近最小值处越来越小，从而使step自动变小。

这种梯度下降法每次要用到整个训练集，因而被称为Batch Gradient Descendent。除了线性回归外，很多其他机器学习问题中也会用到梯度下降法。

### 4 线性回归的梯度下降算法

将梯度下降法运用到线性回归中时，关键在于计算出偏导式。线性回归算法如下：
repeat until convergence: {
	$temp_0 :=\theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})$
	$temp_1 :=\theta_1 - \alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})\cdot x^{(i)}$
	$\theta_0,\theta_1 :=temp_0,temp_1$
}

***

**L3 - 线性代数回顾 Linear Algebra Review (optional)**

***

## L4 - 多变量线性回归 Multivariate Linear Regression

### 1 多变量线性回归

需要考虑多个特征 features 时，采用**多变量线性回归**。特征按$x_1, x_2,...,x_n$标号。

**Notation:**
n: 特征数量
$x^{(i)}$: 训练集中第*i*个样本特征的向量
$x^{(i)}_j$: 训练集第*i*个样本的特征*j*

假设函数变为$h_{\theta}(x)=\theta_0+\theta_1 x_1 + \theta_2 x_2+\cdots+\theta_n x_n$，$x=[x_1,\cdots,x_n]^T$
为了方便表示，我们添加一个$x_0=1$，令$x=[x_0,\cdots,x_n]^T$，$\theta=[\theta_0,\cdots,\theta_n]^T$：
$$
h_{\theta}(x)=\theta^Tx
$$

### 2 多元梯度下降法

在多元的情况下，代价函数为$J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2$，其中$\theta$和$x^{(i)}$是$n+1$维向量.

**多元线性回归算法：**
repeat until convergence: {
	$\theta_j :=\theta_j - \alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})\cdot x_j^{(i)}$
	(simultaneously update $\theta_j$ for $j=0,1,\cdots,n$)
}

### 3 梯度下降算法实用技巧

**3.1 特征缩放**

当不同特征取值范围差异很大时，代价函数的等值线会变得很细长，梯度下降的过程会来回震荡而变得很慢。可以通过**特征缩放** Feature Scaling来使得不同特征的范围相近，从而**加速**梯度下降的过程。通常情况下，缩放到大概$-1\leq x_i \leq 1$即可。

**均值归一化：**
将$x_i$替换为$x_i-\mu_i$使得特征的均值为0。通常情况下可用$x_i:=\frac{x_i-\mu_i}{s_i}$来处理范围，其中$\mu_i$表示第$i$种特征的均值，$s_i$可用$max-min$或标准差表示。

**3.2 学习率与调试方法**

判断梯度下降是否正确运行时，可将每次迭代后$J(\theta)$的结果**绘制**出来，如果是平滑下降的曲线且最终趋于平坦，说明已经收敛。也可设置$\epsilon$，当迭代前后$J(\theta)$的变化小于$\epsilon$时，可视作已经收敛。不过这样的$\epsilon$很难确定，因而观察曲线更加稳妥。

若$J(\theta)$曲线上升或呈现出多个U形相连，很可能是$\alpha$取大了。为了选取合适的$\alpha$，通常从小的值开始每次乘以大约3倍进行测试，例如：..., 0.001, 0.003, 0.01, 0.03, ...。最终得到大小合适的学习率。

### 4 特征选取与多项式回归

**特征选取：**有时候选取合适的特征能够提高模型的表现。例如，计算房价时有frontage和depth两组特征，我们可以直接通过area = frontage $\times$ depth这个特征建立模型。

**多项式回归 polynomial regression：**有些数据集更适合用多项式建立模型，例如$h_{\theta}(x_1)=\theta_0+\theta_1x_1+\theta_2x_1^2+\theta_3x_1^3$。可以令$x_2=x_1^2,x_3=x_1^3$将多项式回归转化为多元线性回归。这里要注意通过**特征缩放**来控制变量范围。

### 5 正规方程

在特征数量$n\leq10000$的情况下，通常采用**正规方程** normal equation method会比梯度下降法快得多。对于**线性回归模型**，正规方程提供了梯度下降法的替代方案。

令$x^{(i)}=[1\,x_1^{(i)}\,\cdots\,x_n^{(i)}]^T$，$X=[x^{(1)}\,x^{(2)}\,\cdots\,x^{(m)}]^T$是一个$m\times (n+1)$的矩阵，$y=[y^{(1)}\,y^{(2)}\,\cdots\,y^{(m)}]$是一个$m$维向量。可计算出最优解:
$$
\theta=(X^TX)^{-1}X^Ty
$$
$X^TX$**不可逆**：极少数情况下$X^TX$会不可逆。通常有两种原因：1.出现了冗余特征，例如同样的数据用不同单位表示，删除冗余特征即可 2.特征数量太多，$m\leq n$可能导致不可逆的情况，删除一些特征或正规化 regularization即可。通常编程语言的库中会提供伪逆函数，使用它可以确保正规方程的正确性。

***

**L5 - Octave教程**

***

## L6 - 逻辑回归 Logistic Regression

### 1 逻辑回归

在**分类问题**中，输出值是离散的。通常输出会分为两类，1为正类positive class，0为负类negative class。这类问题用线性规划是极不合适的，我们希望假设函数的输出能在$[0,1]$之间。分类问题中最常用的算法是**逻辑回归** logistic regression。

**逻辑回归：**线性回归的假设函数是$h_{\theta}(x)=\theta^Tx$，为使范围在$[0,1]$，可采用$h_{\theta}(x)=g(\theta^Tx)$。其中$g(z)=\frac{1}{1+e^{-z}}$是sigmoid函数，也称logistic函数.
$$
h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}
$$
**对假设函数输出的理解**：$h_\theta(x) = P(y=1|x;\theta)$，即在给定的输入$x$和参数$\theta$下，输出$y=1$的概率。由于只有两种输出，$y=0$的概率用1减去即可。

**决策边界**：通常选取$h_\theta(x)\geq0.5$作为$y=1$，$h_\theta(x)<0.5$作为$y=0$。决策边界decision boundary取决于假设函数本身，当$\theta$确定后决策边界也就确定了。复杂的高阶多项式$\theta^Tx$可以刻画出复杂的决策边界。结合sigmoid函数可知，$\theta^Tx=0$即为决策边界。

### 2 代价函数

在线性回归中，我们定义代价函数$J(\theta)=\frac{1}{m}\sum_{i=1}^m\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$。我们不妨定义$Cost(h_\theta(x),y)=\frac{1}{2}(h_\theta(x)-y)^2$，则$J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})$。线性回归中$Cost$函数平方的定义是可行的，但是在逻辑回归中，由于$h_\theta(x)$是非线性函数，通过凸性分析证明最终的$J(\theta)$不是凸函数，会有很多个局部最优解。这样梯度下降法很难得到正确的最优解。因此，我们希望定义$Cost$函数使得$J(\theta)$是一个凸函数。

**逻辑回归代价函数：**
$$
Cost(h_\theta(x),y)=\left\{\begin{matrix}
-log(h_\theta(x))\,\,if\,\,y=1\\ 
-log(1-h_\theta(x))\,\,if \,\,y=0
\end{matrix}\right.\\
$$

$$
J(\theta)=\frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)})
$$

可以发现，当$h_\theta(x)$做出正确的预测时，代价近乎为0；当$h_\theta(x)$做出的预测截然相反时，代价非常大。通过凸性分析可以证明这个代价函数时凸函数。

**代价函数简化：**利用$y$为0或1的特性，代价函数可简化为：
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(h_\theta(x))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]
$$

### 3 梯度下降法

和线性回归一样，剩下的任务只需用梯度下降法将$J(\theta)$最小化

**逻辑回归算法：**
repeat until convergence{
	$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$
	(simultaneously update $\theta_j$ for $j=0,1,\cdots,n$)
}

求偏导后这个算法和线性回归算法非常相似，唯一的不同在于$h_\theta(x)$。另外，线性回归算法中特征缩放等技巧也可运用于逻辑回归。

### 4 高级优化

在大规模机器学习问题中，同样利用$J(\theta)$和$\frac{\part}{\part\theta_j}J(\theta)$，一些更加高级的优化算法具有远好于梯度下降法的表现：

+ 共轭梯度法 Conjugate gradient
+ BFGS
+ L-BFGS

这些算法具有智能的内循环，无需手动挑选$\alpha$且更加高效。高级算法使用起来并不困难，但其内部复杂的原理需要几周去理解。如果不是数值计算方面的专家，建议不要自己手写这些算法，最好直接使用别人写的优质软件库。

### 5 多类别分类：一对多

此前的逻辑回归算法仅能根据输入给出正类或负类，但很多时候我们需要**多类别分类** multi-class classification。一种常用的方法是**一对多** one vs all(one vs rest)：
当用逻辑回归训练类型$i$的分类器时，创建一个伪训练集，类型$i$为1，其他类型为0。最终对于每个类型$i$，得到分类器$h_\theta^{(i)}(x)$，预测出$y=i$的概率。根据$\max_ih_\theta^{(i)}(x)$可以推断出时类型$i$.

***

## L7 - 正则化 Regularization

### 1 过拟合

**过拟合 overfitting：**当特征数量过多时，假设函数可能会很好地拟合训练数据，但无法泛化到新样本上.
欠拟合往往是假设函数与训练数据间存在过大差距 high bias，而过拟合往往因为过多的特征变量或过少的训练数据导致假设函数具有高方差 high variance。在线性回归和逻辑回归中都可能遇到过拟合的情况。

**常见解决方法：**

+ 减少特征数量：

  1. 人工挑选要保留的特征
  2. 使用**模型选择算法**自动选择特征

  这种方法的代价是缺失一些问题相关的信息

+ 正则化：
  保留所有的特征，但是减小参数$\theta_j$的数值或量级
  当特征数量很多且每个特征都对预测$y$有一点影响时，正则化很有效

### 2 正则化与代价函数

**正则化思想：**更小的参数$\theta_0,\theta_1,\cdots, \theta_n$会带来更简洁的假设函数，从而减小过拟合的概率。因此，可对**线性回归的代价函数**进行如下修改：
$$
J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2+\lambda \sum_{j=1}^n\theta_j^2 ]
$$
其中$\lambda$是**正则化参数**。注意$j$的取值从0或1开始皆可，习惯上不考虑$\theta_0$.

**理解：**代价函数中第一项的目标是准确拟合数据，第二项的目标是减小$\theta$从而减轻过拟合的情况。$\lambda$是为了控制这两个目标间的平衡关系，因而选取合适的$\lambda$至关重要。过大的$\lambda$会导致惩罚程度过大，所有参数趋于0，从而欠拟合。后面讲到多重选择时会介绍很多自动选择$\lambda$的方法。

### 3 正则化线性回归

**梯度下降算法：**
repeat {
	$\theta_0 := \theta_0 -\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} $
	$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$      ($j=1,2,\cdots,n$)
}
变化在于$1-\alpha\frac{\lambda}{m}$，注意略它小于1，正则化后$\theta_j$在每次迭代后都会缩小一点。直觉上会发现收敛时$\theta_j$比没有正则化的情况下缩小一些。

**正规方程法：**
$\theta:=(X^TX+\lambda\begin{bmatrix}0 &0 \\ 0 &I_n \end{bmatrix})^{-1}X^Ty$
数学上可以证明正则化后不会出现不可逆的情况。

### 4 正则化逻辑回归

正则化**逻辑回归的代价函数**如下：
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(h_\theta(x))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2
$$
**梯度下降算法：**
repeat{
	$\theta_0 := \theta_0 -\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_0^{(i)} $
	$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m})-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$      ($j=1,2,\cdots,n$)
}
形式与正则化线性回归相同，但是$h_\theta(x)$本质上不同.

**高级优化算法：**
高级优化算法需要计算出正则化的$J(\theta)$和$\frac{\part}{\part\theta_j}J(\theta)$，将计算出的算式输入即可.

***

## L8 - 神经网络: 表示 Neural Networks: Representation

### 1 神经网络模型

**线性回归与逻辑回归的局限：**非线性假设下为了拟合较为复杂的数据，多项式回归中往往需要数量庞大的feature。这种情况下线性回归和逻辑回归的计算代价过大。
在复杂的**非线性假设**下，我们通常选择**神经网络** Neural Networks。神经网络很早就被提出，不过由于计算量庞大，直到近年才大规模应用于机器学习问题。

**神经元模型-逻辑单元 logistic unit：**每个神经元输入$x$，输出$h_\theta(x)=\frac{1}{1+e^{-\theta^Tx}}$。这里的Sigmoid函数是神经网络的**激活函数** activation function，$\theta$是参数，也称模型的**权重**。绘制模型时，输入可省去**偏置单元** bias unit: $x_0=1$。

神经网络的第一层为**输入层** input layer，用于输入特征；最后一层为**输出层** output layer，用于输出计算结果；其余层为**隐藏层** hidden layer。
不同的**神经网络架构** network architecture中神经元的连接方式不同。

**Notation:**
$a_i^{(j)}$：第$j$层第$i$个单元的**激活项**，即计算并输出的值
$\Theta^{(j)}$：控制第$j$层到第$j+1$层映射函数的**权重矩阵**。若第$j$层有$s_j$个单元，第$j+1$层有$s_{j+1}$个单元，则权重矩阵的维数为$s_j\times(s_{j+1}+1)$

**神经网络与逻辑回归：**神经网络每个神经元与上一层激活项间的关系实质上就是逻辑回归与其输入。不同点在于，神经网络每一层的输入不是模型输入的特征$x_1,x_2,x_3\cdots$，而是上一层输出的$a_1^{(j)},a_2^{(j)},a_3^{(j)}\cdots$，根据为权重矩阵选取的不同参数，可以学习到一些更为有趣且复杂的特征。

### 2 前向传播

**前向传播 Forward Propagation：**计算从输入层的激活项开始，逐层传播，最终传播到输出层.

**神经网络的计算过程：**考虑Layer 2，假设输入层有$x_1, x_2, x_3$
$a^{(2)}_1=g(\Theta_{10}^{(1)}x_0+\Theta_{11}^{(1)}x_1+\Theta_{12}^{(1)}x_2+\Theta_{13}^{(1)}x_3)=g(z^{(2)}_1)$
$a^{(2)}_2=g(\Theta_{20}^{(1)}x_0+\Theta_{21}^{(1)}x_1+\Theta_{22}^{(1)}x_2+\Theta_{23}^{(1)}x_3)=g(z^{(2)}_2)$
$a^{(2)}_3=g(\Theta_{30}^{(1)}x_0+\Theta_{31}^{(1)}x_1+\Theta_{32}^{(1)}x_2+\Theta_{33}^{(1)}x_3)=g(z^{(2)}_3)$
令$x=[x_0\,x_1 \,x_2\,x_3]^T$，$z^{(2)}=[z^{(2)}_1\,z^{(2)}_2\,z^{(2)}_3]^T$
**计算向量化：**
$z^{(2)}=\Theta^{(1)}a^{(1)}$ # 用权重矩阵和Layer 1激活项计算向量$z^{(2)}$
$a^{(2)}=g(z^{(2)})$ # 用$z^{(2)}$计算出Layer 2激活项向量，加上$a^{(2)}_0=1$偏置单元
$z^{(3)}=\Theta^{(2)}a^{(2)}$ # 用权重矩阵和Layer 2激活项计算向量$z^{(3)}$
$a^{(3)}=g(z^{(3)})$ # 用$z^{(3)}$计算出Layer 3激活项向量，加上$a^{(3)}_0=1$偏置单元
$\cdots\cdots$ # 前向传播
$h_\theta(x)=a^{(m)}=g(z^{(m)})$ # 输出层

### 3 多类别分类

神经网络的输出层设置多个单元可实现多类别分类，这实质上是一对多方法的拓展。训练集中数据为$(x^{(i)},y^{(i)})$，其中$y^{(i)}$不再是$1,2,3\cdots$，而是$[1\,0\,0\,\cdots]^T,$$[0\,1\,0\,\cdots]^T,$$[0\,0\,1\,\cdots]^T$中的一个。

***

## L9 - 神经网络: 学习 Neural Networks: Learning

### 1 代价函数

神经网络的代价函数基于逻辑回归的代价函数，我们重点关注神经网络分类问题的代价函数。

**二元分类：**$K=1$，仅一个输出单元，输出0或1
**多元分类：**$K\geq3$，有$k$个输出单元，输出$y=h_\Theta(x)\in\mathbb{R}^k$

**Notation：**
$\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$：训练集
$L$：神经网络层数
$s_l$：第$l$层除偏置单元外的神经元数量
$(h_\Theta(x))_i$：神经网络的第$i$个输出

基于正则化逻辑回归的代价函数$J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^my^{(i)}\log(h_\theta(x))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]+\frac{\lambda}{2m}\sum_{j=1}^n\theta_j^2$，
可以得到**神经网络的代价函数**：
$$
J(\Theta)=-\frac{1}{m} \left [\sum_{i=1}^m\sum_{k=1}^Ky_k^{(i)}\log(h_\theta(x))_k+(1-y^{(i)}_k)\log(1-(h_\theta(x^{(i)}))_k)\right]\\+\frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{ji}^{(l)})^2
$$

这个代价函数是正则化的，习惯上$i$从1开始取值。

### 2 反向传播算法

为了找到参数$\Theta$来最小化，需要使用梯度下降法或其他的高级优化算法。因此，我们需要计算$J(\Theta)$和$\frac{\part}{\part\Theta^{(l)}_{ij}}J(\Theta)$，我们已经有了前者，现在重点关注**偏导项**。

**梯度计算 - 前向传播**
$a^{(1)}=x$
$z^{(2)}=\Theta^{(1)}a^{(1)}$
$a^{(2)}=g(z^{(2)})$ (add $a^{(2)}_0$)
$z^{(3)}=\Theta^{(2)}a^{(2)}$
$a^{(3)}=g(z^{(3)})$ (add $a^{(3)}_0$)
$\cdots\cdots$
$z^{(L)}=\Theta^{(L-1)}a^{(L-1)}$
$a^{(L)}=h_\Theta(x)=g(z^{(L)})$ 
**梯度计算 - 反向传播**
$\delta^{(l)}_j$：第$l$层第$j$个神经元的误差项
输出层：$\delta^{(l)}=a^{(L)}-y$
其余层：$\delta^{(j)}=(\Theta^{(j)})^T\delta^{(j+1)}.*g'(z^{(j)})=(\Theta^{(j)})^T\delta^{(j+1)}.*[a^{(j)}.*(1-a^{(j)})]$
其中.*为pairwise operation。由于从输出层向前计算，我们称之为反向传播。$\delta^{(1)}$无需计算，因为输入层不存在误差。另外，可以证明在不考虑$\lambda$或$\lambda=0$的情况下，$\frac{\part}{\part\Theta^{(l)}_{ij}}J(\Theta)=a^{(l)}_j\delta_i^{(l+1)}$.

**反向传播算法 Backpropagation algorithm：** {
Training Set $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$
Set $\Delta^{(l)}_{ij}=0$ for all $l,i,j$ 
For $i=1$ to $m$
	Set $a^{(1)}=x^{(i)}$
	Perform forward propagation to compute $a^{(l)}$ for $l=2,3,\cdots,L$
	Using $y^{(i)}$, compute $\delta^{(l)}=a^{(L)}-y^{(i)}$
	Perform backprop to compute $\delta^{(l)}$ for $l=L-1,L-2,\cdots,2$
	$\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T$ for $l=1,2,\cdots,L-1$

$D_{ij}^{(l)}:=\frac{1}{m}\Delta^{(l)}+\lambda\Theta_{ij}^{(l)}$   if $j\neq 0$
$D_{ij}^{(l)}:=\frac{1}{m}\Delta^{(l)}$   if $j = 0$
}

这个算法目标是计算出$\frac{\part}{\part\Theta^{(l)}_{ij}}J(\Theta)$，计算结果保存在$D_{ij}^{(l)}$内。遍历训练集的for循环内，先用前向传播计算出激活项$a^{(l)}$，然后用后向传播计算出误差项$\delta^{(l)}$，最后将计算结果累加到$\Delta^{(l)}$内。循环结束后，根据$\Delta^{(l)}$的结果并考虑正则项，计算出$D^{(l)}_{ij}$。

### 3 梯度检验

在复杂的计算模型中，**反向传播算法**或相似的梯度下降算法实现时很容易出现bug。有时整个过程表现得很正常，$J(\Theta)$会逐渐下降到一个最小值，但最终模型仍有很大的误差。**梯度检验** gradient checking可以大幅降低出错得可能性，在这类模型中建议使用梯度检验来确保代码的正确性。

**梯度检验原理：**利用**双侧差分**计算出导数近似值，将近似值与反向传播的结果比较.
(双侧差分$\frac{\part}{\part\theta}J(\theta)\approx\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$的准确性高于单侧差分$\frac{J(\theta+\epsilon)-J(\theta)}{\epsilon}$，通常$\epsilon=10^{-4}$即可)

**梯度检验：**将神经网络中的所有参数$\Theta^{(l)}$展开成一个长向量$\theta=[\theta_1,\theta_2,\cdots,\theta_n]$，接下来验证：
$\frac{\part}{\part\theta_1}J(\theta)\approx\frac{J(\theta_1+\epsilon,\theta_2,\cdots,\theta_n)-J(\theta_1-\epsilon,\theta_2,\cdots,\theta_n)}{2\epsilon}$
$\frac{\part}{\part\theta_2}J(\theta)\approx\frac{J(\theta_1,\theta_2+\epsilon,\cdots,\theta_n)-J(\theta_1,\theta_2-\epsilon,\cdots,\theta_n)}{2\epsilon}$
$\cdots\cdots$
$$\frac{\part}{\part\theta_n}J(\theta)\approx\frac{J(\theta_1,\theta_2,\cdots,\theta_n+\epsilon)-J(\theta_1,\theta_2,\cdots,\theta_n-\epsilon)}{2\epsilon}$$
式子左边为反向传播的计算结果，右边为双侧差分计算出的近似值.

**反向传播与梯度检验的实现：**
1.反向传播计算
2.梯度检验计算
3.比较计算结果
4.若结果无误，**关闭梯度检验**进行训练(梯度检验的计算代价很高)

### 4 随机初始化

梯度下降法或高级优化算法需要对参数$\theta$初始化。在神经网络模型中，把参数全部初始化为0会造成高度的冗余。如果权重矩阵每行相同，则迭代后每层的神经元完全相同：它们接受同样的参数，有相同的$\alpha^{(l)}_j$和$\delta^{(l)}_j$。实际上最后每个逻辑回归单元仅得到一个特征。这就是**对称权重问题**，可通过随机初始化解决。

通常将$\Theta^{(l)}_{ij}$初始化为$[-\epsilon,\epsilon]$间的随机值，然后再进行训练。

### 5 神经网络实践

**神经网络架构选择：**
输入单元数量：特征$x^{(i)}$的维度
输出单元数量：类型数
默认架构：大多数情况下，一个隐藏层是最好的选择。如果有多余一个隐藏层，每个隐藏层的神经元数量应该相同。隐藏神经元的数量越多则效果越好，但是计算量也会更大。

**神经网络训练：**

1. 对权重矩阵随机初始化
2. 对训练集中的数据，用前向传播计算出$h_\Theta(x^{(i)})$
3. 计算代价函数$J(\Theta)$
4. 反向传播计算出偏导项$\frac{\part}{\part\Theta_{ij}^l}J(\Theta)$
5. 用梯度检验检查反向传播和数值估计的结果，无误后关闭梯度检验
6. 用梯度下降法或高级优化算法，结合反向传播算法，最小化$J(\Theta)$

PS：神经网络的代价函数$J(\Theta)$不是凸函数，理论上使用梯度下降法和高级优化算法等方法的结果可能是局部最优解，但是实践中这样的结果通常已经足够好。

***

## L10 - 应用机器学习的建议 Advice for Applying ML

### 1 机器学习诊断

在模型运行效果与预期相差甚远时，人们通常随机选择下列的某个方法，投入大量时间调试机器学习算法：

+ 使用更多的训练样本
+ 减少/增加特征数量
+ 增加多项式特征数量
+ 增加/减小正则化参数$\lambda$

在大规模项目中，这样盲目的调试可能会浪费很多时间。

**机器学习诊断 Machine Learning Diagnostic：**通过测试来发现机器学习哪一部分出现了问题，获得提高性能方面的指导，确定接下来应该做什么

### 2 评估假设函数

为了评估模型，通常将数据集按照7:3的比例分为**训练集** training set和**测试集** test set。为了衡量假设函数$h_\theta(x)$的表现，需要定义**测试误差** test set error。

**线性回归测试误差：**
$$
J_{test}(\theta)=\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_\theta(x_{test}^{(i)})-y_{test}^{(i)})^2
$$
**逻辑回归测试误差：**
$$
J_{test}(\theta)=-\frac{1}{m_{test}}\sum_{i=1}^{m_{test}}y_{test}^{(i)}\log h_\theta(x_{test}^{(i)})+(1-y_{test}^{(i)})\log h_\theta(x_{test}^{(i)})
$$
逻辑回归的测试误差还有一个较直观的量度：**错误分类 misclassification error**。定义$err(h_\theta(x),y)=[h_\theta\geq0.5,y=1\,or\,h_\theta<0.5,y=0]$，这里用到艾弗森括号。
$$
Test\,\,error=\frac{1}{m_{test}}\sum_{i=1}^{m_{test}}err(h_\theta(x_{test}^{(i)}),y_\theta^{(i)})
$$

### 3 模型选择与训练/验证/测试集

**模型选择问题：**多项式次数、正则化参数$\lambda$等会影响模型的最终表现，通常需要比较不同模型的表现选择最优的模型。

在选择模型时，通常间数据集按照6:2:2的比例划分为训练集、**(交叉)验证集** cross validation set和**测试集**。这三组数据分别产生如下误差：
**训练误差**：$J_{train}(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$
**验证误差**：$J_{cv}(\theta)=\frac{1}{2m_{cv}}\sum_{i=1}^{m_{cv}}(h_\theta(x_{cv}^{(i)})-y_{cv}^{(i)})^2$
**测试误差**：$J_{test}(\theta)=\frac{1}{2m_{test}}\sum_{i=1}^{m_{test}}(h_\theta(x_{test}^{(i)})-y_{test}^{(i)})^2$
对多个模型选择进行训练后，通过验证集找出$J_{cv}(\theta)$最小的模型。假设不同的模型区别在于多项式次数$d$，那么通过验证集筛选模型的过程实质上是对$d$进行了一次拟合。为了确保模型**泛化误差**较小，还需要通过测试集进行测试。有时将验证集和训练集合并，即用筛选模型的数据集进行测试，也可以得到泛化误差很小的模型，但是最好将其分开。

### 4 正则化与偏差、方差

机器学习的问题通常归为两类：**偏差 bias**较大(欠拟合)和**方差 variance**较大(过拟合)。分清是那种情况对于解决问题十分重要。

考虑多项式次数$d$，训练误差$J_{train}(\theta)$会随着$d$的增大而减小，验证误差$J_{cv}(\theta)$会随着$d$的增大而先减后增。因而可以通过如下方式判断：

+ 偏差：$J_{train}(\theta)$较大
  	$J_{train}(\theta)\approx J_{cv}(\theta)$
+ 方差：$J_{train}(\theta)$较小
      $J_{train}(\theta)\ll J_{cv}(\theta)$

偏差、方差也常与正则化参数$\lambda$相关。过大的$\lambda$会导致高偏差，过小的$\lambda$会导致高方差。与代价函数$J(\theta)$不同的是，$J_{train}(\theta),J_{cv}(\theta),J_{test}(\theta)$中不考虑正则化项。$J_{train}(\theta)$会随着$\lambda$的增大而增大，$J_{cv}(\theta)$会随着$\lambda$的增大而先减后增，通过这种判断可以找到合适的$\lambda$.

### 5 学习曲线

**学习曲线 learning curves**可以有效判断是高偏差还是高方差。学习曲线横轴为训练集规模$m$，纵轴为误差$error$。

**正常情况：**$J_{train}(\theta)$随着训练的规模增大而逐渐增大，$J_{cv}(\theta)$随着训练的规模增大而逐渐变小，两者的差距不大且逐渐缩小。
**高偏差：**$J_{train}(\theta)$随着训练规模的增大而快速上升，达到一定规模后几乎平行于横轴。$J_{cv}(\theta)$随着训练规模增大而逐渐下降，达到一定规模后也趋于平行，且和$J_{train}(\theta)$差距非常小。由此可见，增加训练样本数量不能解决高偏差的情况。
**高方差：**$J_{train}(\theta)$随着训练规模的增大而逐渐增大，$J_{cv}(\theta)$随着训练的规模增大而逐渐变小，但两者差距始终较大。从趋势上来看，两者最终会接近，因而增加训练样本可以改善高方差的情况。

实践中曲线可能有较多的噪声，但是总体趋势通常是符合上文描述的。

### 6 解决问题

了解了如何诊断出高偏差和高方差的问题后，可以针对问题采用不同方案，避免盲目尝试造成的时间浪费。

+ 使用更多的训练样本 -> 高方差
+ 减少特征数量 -> 高方差
+ 增加特征数量 -> 高偏差
+ 增加多项式特征数量 -> 高偏差
+ 增加正则化参数$\lambda$ -> 高方差
+ 减小正则化参数$\lambda$ -> 高偏差

另外，针对神经网络也有一些建议：小规模的神经网络计算代价小，但容易欠拟合。大规模的神经网络计算代价大，容易过拟合，但是通过正则化通常可以有更好的性能。神经网络的隐藏层数量通常设为1，若需要更多隐藏层，可通过验证集进行模型选择。

***

## L11 - 机器学习系统设计 Machine Learning System Design

### 1 误差分析

在实现一个大型的机器学习项目前，建议先实现一个简单的算法，对其进行**误差分析 error analysis**并通过**数值评估 numerical evaluation**评价表现：

+ 手动分析验证集中**出错样本**的共性，寻找解决方案
+ 通过一个**数值度量标准**衡量每个模型的表现

### 2 不对称分类的误差评估

**偏斜类 skewed classes**是指分类问题中正样本与负样本数量比较十分悬殊。如果采用一般的数值评估方法，即计算出测试样本的误差率，无法正确反映模型的表现。例如，某一模型的准确度为99%，看上去很不错了。但是考虑只有0.5%的样本是正类，这样的表现就不那么好了。相比之下，如果一个模型永远输出$y=0$，准确度为99.5%，甚至高于一个训练过的模型，显然它的预测准确度不能反映其表现。

分类问题中，预测结果与实际分类的组合如下：

| predicted\actual | 1              | 0              |
| :--------------: | -------------- | -------------- |
|        1         | True Positive  | False Positive |
|        0         | False Negative | True Negative  |

在不对称分类的情况下，通过**查准率 precision**和**召回率 recall**来衡量模型表现.

**查准率：**$\frac{true\,\,pos}{predicted\,\,pos}=\frac{true\,\,pos}{true\,\,pos+false\,\,pos}$
查准率是预测为1的样本中真正为1的比例
**召回率：**$\frac{true\,\,pos}{actual\,\,pos}=\frac{true\,\,pos}{true\,\,pos+false\,\,neg}$
召回率是真正为1的样本中预测出为1的比例

### 3 查准率与召回率的权衡

通常分类器需要在查准率和召回率之间权衡。逻辑回归中在$h_\theta(x)\geq threshold$时预测$y=1$，在$h_\theta(x)<threshold$时预测$y=0$。阈值越高，预测1时越有把握，查准率越高，召回率越低；相反，阈值越低，预测0时越有把握，查准率越低，召回率越高。

由于这两个度量标准此消彼长， 不同的算法有时很难直接根据它们进行取舍。我们定义**F值**($F_1$ score)，$F=2\frac{PR}{P+R}$。这样的定义给召回率和查准率中较低的那个更高的权重，同时确保取值范围在$[0,1]$之间。

### 4 机器学习数据

关于机器学习的数据可以考虑以下两个方面：

1. 人类专家应该能够从提供给算法的数据$x$中预测$y$
2. 算法有足够的参数，数据有足够的特征与规模

在上述条件下，大多数算法都可以有相似的表现。

***

## L12 - 支持向量机 Support Vector Machines

### 1 优化目标

逻辑回归中代价函数是$J(\theta)=-\frac{1}{m}\sum_{i=1}^m[y\log(h_\theta(x))+(1-y)\log(1-h_\theta(x))]$，考虑其中的式子$-y\log\frac{1}{1+e^{-\theta^Tx}}-(1-y)\log(1-\frac{1}{1+e^{-\theta^Tx}})$。
如果$y=1$，期待$\theta^Tx\geq 0$。此时上式变为$-\log\frac{1}{1+e^{-\theta^Tx}}$，将$z=\theta^Tx$视为变量，定义函数$cost_1(z)$，该函数在$z\geq 1$时为0，$z<1$是一条斜率为负的直线，与上式贴近；
同理，$y=0$时期待$\theta^Tx<0$。上式变为$-\log(1-\frac{1}{1+e^{-\theta^Tx}})$。定义函数$cost_0(x)$，该函数与$cost_1(x)$关于$y$轴对称，也与上式贴近。

在此基础上，再对逻辑回归代价函数作如下修改：
1.两边同时乘以常数$m$，这不会影响最优的参数$\theta$；
2.取消正则化参数，在第一项前添加常数$C$来平衡两项的关系.

由此可以得到**支持向量机 SVM**的**优化目标：**
$$
\min_\theta C\sum_{i=1}^m\left [y^{(i)}cost_{1}(\theta^Tx^{(i)})+(1-y^{(i)})cost_0(\theta^Tx^{(i)})\right]+\frac{1}{2}\sum_{i=1}^n\theta_j^2
$$
SVM的**假设函数**为：
$$
h_\theta(x)=
\left\{\begin{matrix}
1\,\,\,\,\,\,\,\,\,\,\,\,\,\,if\,\,\theta^Tx\geq0\\ 
0\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,otherwise
\end{matrix}\right.\\
$$

### 2 SVM大间隔

由于$cost_1,cost_0$函数的特点，发现SVM希望$y=1$时$\theta^Tx\geq1$，$y=0$时$\theta^T\leq-1$。在$C$非常大的情况下，优化目标第一项应变为0，此时优化问题等价于：

+ 计算$\theta$最小化$\frac{1}{2}\sum_{i=1}^n\theta_j^2$，满足$\theta^Tx\geq1$ if $y=1$，$\theta^Tx\leq-1$ if $y=0$.

这个优化问题会使得SVM不仅能完成分类，还会选择与两边数据有最大**间隔** margin的决策边界，因此SVM也被称为**大间隔分类器** large margin classifier。这种特性增强了SVM分类器的鲁棒性。

实践中，过大的$C$会导致SVM对异常数据敏感，从而影响间隔。合适大小的$C$能帮助SVM提高对异常数据的耐受程度。

**数学原理：**
从向量的角度，$\frac{1}{2}\sum_{i=1}^n\theta_j^2=\frac{1}{2}\left\|\theta\right\|^2$，$\theta^Tx^{(i)}=p^{(i)}\left\|\theta\right\|$，其中$p^{(i)}$是$x^{(i)}$在$\theta$上的投影。另外，$\theta^Tx=0$意味着$\theta$与决策边界正交。若SVM的决策边界的间隔很小，会存在一些样本在$\theta$上的投影很小，为了满足$p^{(i)}\left\|\theta\right\|\geq1$或$p^{(i)}\left\|\theta\right\|\leq-1$，$\|\theta\|$会因此变得很大，这与优化目标恰好相反。因此，优化后的决策边界会与尽量与样本间保持大间隔。注意这个优化问题的前提是在$C$很大。

### 3 核函数

为了得到复杂的非线性函数，通常会选取复杂的多项式$\theta_0+\theta_1f_1+\theta_2f_2+\theta_3f_3+\cdots$，其中$f_1=x_1,f_2=x_2,f_3=x_1x_2,f_4=x_1^2,f_5=x_2^2\cdots$
这种方法会面临大量的特征选择，复杂问题中计算量很大。我们需要一个找到更好的特征$f_1,f_2,f_3\cdots$的方法。

选取一些**标记 landmarks**：$l^{(1)},l^{(2)},l^{(3)},\cdots$，给定$x$向量时，可以计算$x$与这些标记的接近程度，从而得到新的特征$f_1,f_2,f_3,\cdots$.
$$
f_i=similarity(x, l^{(i)})=\exp(-\frac{\left\|x-l^{(i)} \right\|}{2\sigma^2})=\exp(-\frac{\sum_{j=1}^n(x_j-l^{(i)}_j)^2}{2\sigma^2})
$$
这里similarity是相似度函数，即**核函数 kernel**，可写作$k(x,l^{(i)})$。本式中核函数选取的是**高斯核函数 Gauss kernel**，$x$与$l^{(i)}$接近时近似于1，距离较远时近似于0。其中参数$\sigma$越大则下降越慢，反之下降越快。核函数得到的特征能够帮助SVM有效地学习复杂的特征。

**标记选取：**通常我们将训练样本都作为标记，即$l^{(1)}=x^{(1)},l^{(2)}=x^{(2)},\cdots,l^{(m)}=x^{(m)}$. 这样给定$x$后，可得$f_1=k(x,l^{(1)}),f_2=k(x,l^{(2)}),\cdots,f_m=k(x,l^{(m)})$，添加$f_0=1$.

**SVM与核函数：**
给定$x$，计算出特征$f\in\mathbb{R}^{m+1}$，若$\theta^Tf\geq0$，预测$y=1$
训练时，最小化$C\sum_{i=1}^m\left [y^{(i)}cost_{1}(\theta^Tf^{(i)})+(1-y^{(i)})cost_0(\theta^Tf^{(i)})\right]+\frac{1}{2}\sum_{i=1}^m\theta_j^2$.
SVM与核函数结合有很多提高计算效率的方法，例如将$\theta$的平方和转化为$\theta^TM\theta$来计算，其中$M$与核函数的选取有关。这里的计算方法涉及到很多数学细节，最好直接用软件库优化目标。另外，核函数原理上也可以应用于逻辑回归等其他算法，但是由于一些计算优化无法推广到逻辑回归中，核函数会使得其他算法变慢很多。

**SVM参数：**
$C$：$C$过大时会导致高方差，过小时会导致高偏差.
$\sigma^2$：较大的$\sigma^2$会使特征$f_i$变化较为平滑，可能导致高偏差；较小的$\sigma^2$会使特征$f_i$变化较为剧烈，可能导致高方差.

### 4 SVM应用

**核函数：**SVM可以不使用核函数(或者叫线性核函数 linear kernel)。使用核函数时，除了高斯核函数外，还有很多复杂的核函数可供选择：polynomial kernel, string kernel, chi-square kernel, histogram intersection kernel, ...

**多元分类：**许多SVM软件包已经实现了多元分类的功能，当然也可用一对多方法实现多元分类。

**算法选择：**

+ **特征数相比于样本数很大**，选择逻辑回归/SVM+线性核函数
  这种情况下线性函数已经有很好的表现，这样的数据量也难以训练出复杂的函数
+ **特征少，样本数量适中**，选择SVM+高斯核函数
+ **特征少，样本数量大**，添加特征并选择逻辑回归/SVM+线性核函数
  尽管SVM的核函数在是线上已经优化了很多，但是在庞大的数据量面前，SVM+高斯核函数还是会很慢

神经网络适用于大多数情形，但是训练起来较慢。此外，SVM的优化问题处理的是凸函数，不会遇到局部最优的问题，这对神经网络来说是个不大不小的问题。
此外，逻辑回归和SVM+线性核函数尽管有时在表现上有差异，但是应用场景时互通的。

***

## L13 - 聚类 Clustering

### 1 无监督学习

监督学习中，训练集有标签：$\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots,(x^{(m)},y^{(m)})\}$. 但在无监督学习中训练集变为$\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$，算法需要找到隐含在数据中的结构。例如，在数据中找到几组**簇 clusters**，这类算法被称为**聚类算法 clustering algorithm.**

### 2 K-means算法

**K-means算法**是一种应用广泛的聚类算法，将数据分为K个簇.

算法首先随机选取$K$个**聚类中心 cluster centroids**，加下来迭代。每次迭代分为两步：首先是**簇分配 cluster assignment**，将训练集中所有点划分到与之最近的聚类中心。接下来是**移动聚类中心 move centroids**，将所有分配到某聚类中心的点求平均，得到聚类中心的新位置。

**K-means Algorithm:**
Input:
-K (number of clusters)
-Training Set $\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$ (**drop** $x_0=1$ convention)

Randomly initialize $K$ cluster centroids $\mu_1,\mu_2,\cdots,\mu_K\in \mathbb{R^n}$
Repeat \{
	for $i=1$ to $m$ 
		$c^{(i)} :=$ index (1~$K$) of cluster centroid closest to $x^{(i)}$ 
	for $k=1$ to $K$ 
		$\mu_k:=$ mean of points assigned to cluster $k$
\}

有些应用场景下数据没有清晰地分成若干簇，此时K-means算法仍然可以应用。另外，若训练时某个聚类中心没有分配到任何点，通常的做法是将其删除，有时也可考虑重新随机分配聚类中心进行训练。

### 3 优化目标

**Notation:**
$c^{(i)}$：$x^{(i)}$被分配到的簇的下标
$\mu_k$：下标为$k$的簇 ($\mu_k\in\mathbb{R}^n$)
$\mu_{c^{(i)}}$：$x^{(i)}$被分配到的簇

和线性回归、SVM、神经网络等算法一样，K-means算法有优化目标：
$$
\min_{c^{(1)}\cdots c^{(m)},\\\mu_1\cdots \mu_K}J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)=\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-\mu_{c^{(i)}} \right\|^2
$$
K-means算法中簇分配实际上是假定$\mu_1,\cdots,\mu_K$固定，最小化代价函数$J(\cdots)$，移动聚类中心是假定$c^{(1)},\cdots,c^{(m)}$固定，最小化代价函数$J(\cdots)$.

### 4 随机初始化

K-means中一个很好的初始化方法是：随机挑选$K$**个训练样本**设为聚类中心.

**局部最优解：**K-means算法可能获得局部最优解。在K较小的情况下，可以通过多次(通常50~1000次)随机初始化的方式获得一个最好的局部最优解：
For $i=1$ to 100 {
	Randomly initialize K-means
	Run K-means, get $c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K$.
	Compute cost function $J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)$
}
Pick clustering that gave lowest $J(c^{(1)},\cdots,c^{(m)},\mu_1,\cdots,\mu_K)$.
对于较大的K，多次随机初始化作用通常不太显著.

### 5 聚类数量

目前没有很好的自动化方法来决定聚类数量$K$，通常需要手动选择$K$。

**肘部法则 Elbow method：**增加$K$观察$J(\cdots)$，后者下降速度由快突然转慢的地方，即“肘部”，或许是很好的选择。但是有时绘制出的曲线无法清晰观察到一个“肘部”，这是肘部法则不再适用。

另外，很多时候需要根据后续目的(商业目的等)来确定聚类数量。

***

## L14 - 降维 Dimensionality Reduction

### 1 降维目标

**数据压缩：**将数据从3D降为2D，或从2D将为1D，这样可以减小数据的内存或磁盘空间，并且可以提高学习的速度.

**可视化：**高维数据降位低维以便可视化.

### 2 主成分分析 PCA

**主成分分析 Principle Component Analysis**是一种常用的降维算法，其核心思想是在高维空间中寻找低维平面，以最小化**投影误差 projection error**，即每个点与其投影点距离的平方和。

**数据预处理：**
给定训练集$\{x^{(1)}, x^{(2)}, \cdots,x^{(m)} \}$
首先**均值归一化**，计算$\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$，用$x^{(i)}_j-\mu_j$替代$x_j^{(i)}$；
如果有必要，可以进行**特征缩放**，$x_j^{(i)}=\frac{x_j^{(i)}-\mu_j}{s_j}$.

**PCA算法：**PCA算法目标是将$n$维向量降到$k$维。首先计算出样本的**协方差矩阵 covariance matrix**，$\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)})(x^{(i)})^T$。接着用**奇异值分解SVD**(或其他方法)计算出$\Sigma$的特征向量，计算得到的特征向量按照列向量的形式存放在矩阵$U\in\mathbb{R}^{n\times n}$中。取$U$的前$k$列得到$U_{reduce}\in\mathbb{R}^{n\times k}$，则可将$x\in\mathbb{R}^n$降维：$z=U_{reduce}^Tx$，其中$z\in\mathbb{R}^k$。说明如此得到的低维平面具有最小的投影误差需要复杂的数学证明，在此不详细展开。

**PCA Algorithm：**
Compute covariance matrix: $\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)})(x^{(i)})^T$
Compute eigenvectors of matrix $\Sigma$: $U$, obtain $U_{reduce}$
$z = U_{reduce}^Tx$

**压缩重现：**$x_{approx}^{(i)}=U_{reduce}z^{(i)}$，通过这种方式可以将压缩后的数据重现为原高维数据的近似值。

### 3 主成分数量

PCA中需要选择主成分的数量$k$，这时需要考虑两个量.
**平均投影误差** average squared projection error: $\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2$
**全变分** total variation: $\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2$

通常，选择满足下式的最小的$k$：
$$
\frac{\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2}\leq0.01
$$
这里0.01也可换成0.05之类的数据。将0.01理解为：保留99%的variance。

显然上式计算量很大。之前在用SVD方法求$U$时，还会得到奇异值矩阵$S$，$S$是一个$n\times n$矩阵，除对角线元素外都是0。可以得到下列关系：
$$
\frac{\frac{1}{m}\sum_{i=1}^m\left\| x^{(i)}-x_{approx}^{(i)} \right\|^2}{\frac{1}{m}\sum_{i=1}^m\|x^{(i)} \|^2}=1-\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^m S_{ii}}
$$
这样，我们就可以快速用$\frac{\sum_{i=1}^kS_{ii}}{\sum_{i=1}^m S_{ii}}\geq 0.99$找到$k$。

### 4 PCA应用

**加速学习：**通过将高维$x^{(i)}$降为低维$z^{(i)}$，可以获得特征更少的训练集$\{ (z^{(1)},y^{(1)}),(z^{(2)},y^{(2)}),\cdots,(z^{(m)},y^{(m)}) \}$。注意应该只在训练集上运行PCA，得到的$x\rightarrow z$的映射中考虑了均值归一化、特征缩放和$U_{reduced}$等。这个映射在$x_{cv}^{(i)}$和$x^{(i)}_{test}$上也能正常应用。这里PCA要注意对原信息的保留率。

**可视化：**将数据降为2D或3D可以可视化。

**不建议的应用场景：**不应用PCA减少特征数来降低过拟合的可能性。PCA训练时不考虑标签$y^{(i)}$，可能会丢失一些信息。另外，即使保留了95%或99%的variance，通常正则化的效果也至少和PCA一样。
另外，在设计项目时不建议直接使用PCA处理数据，除非有充分的证据表明用$x^{(i)}$无法运行。建议在使用PCA之前先用原数据进行训练。

***

## L15 - 异常检测 Anomaly Detection

### 1 异常检测

**模型描述：**给定数据集$\{x^{(1)},x^{(2)},\cdots,x^{(m)} \}$，判断$x_{test}$是否异常。建立一个概率模型$p(x)$，当$p(x)<\epsilon$时视为异常。

**异常检测**中要用到**高斯分布**：$x\sim \mathcal{N}(\mu,\sigma)$，可算出$p(x;\mu,\sigma)=\frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(x-\mu)^2}{2\sigma^2})$. 对每个特征求高斯分布后，将所有的$p(x_j;\mu_j,\sigma_j)$求积即可。

**异常检测算法：**
Choose features $x_i$ that you think might be indicative of anomalous examples.
Fit parameters $\mu_1,\cdots,\mu_n,\sigma_1^2,\cdots,\sigma^2_n$.
	\- $\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$
	\- $\sigma_j^2=\frac{1}{m}\sum_{i=1}^m(x_j^{(i)}-\mu_j)^2$
Given new examples $x$, compute $p(x)$:
	\- $p(x)=\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)=\prod_{j=1}^n \frac{1}{\sqrt{2\pi}\sigma_j}\exp{(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2})}$ 
Anomaly is $p(x)<\epsilon$

### 2 模型评估

为了评估异常检测模型的表现，需要将数据集进行划分。数据集中包含正常样本和异常样本，通常前者远多于后者。将正常样本按照6:2:2划分到训练集、验证集和测试集，异常样本分到验证集和测试集。训练集$\{x^{(1)},\cdots,x^{(m)} \}$没有标签，验证集$\{x_{cv}^{(1)},\cdots,x_{cv}^{(m_{cv})} \}$和测试集$\{x_{test}^{(1)},\cdots,x_{test}^{(m_{test})} \}$有标签，1表示异常，0表示正常。

数值评价标准可选取查准率、召回率以及$F_1$-Score。另外，异常检测的$\epsilon$也可通过验证集来选取。

### 3 异常检测与监督学习

**异常检测：**
\- 正类样本较少(0-20很常见)，负类样本很多
\- 异常种类较多，且样本不足以支持算法学习到正类的特征，未来的正类可能和现有正类差异很大，这些情况下考虑异常检测

**监督学习：**
\- 正类样本和父类样本都很多
\- 样本足以让算法学习到正类的特征，且未来的正类和现有正类较为相似，则可以考虑监督学习

### 4 特征选取

**非高斯分布特征：**选择不符合高斯分布的特征通常也能使算法运行，但是有一些方法可以将特征转化为近似高斯分布，例如$\log x$, $x^\alpha$等。

**误差分析：**当某个异常样未被识别出时，可专门分析它，找出其与众不同的特征，并将该特征加入算法内。

**特征选取：**通常再异常情况下变化幅度大的特征(特别小或特别大)非常合适，可以再原特征中选取，用$\frac{a}{b}$等方式组合出新特征用于训练。

### 5 多变量高斯分布

有时样本的特征$x_1$和$x_2$都在正常范围内，但是其组合是异常的，这是因为$x_1$和$x_2$有相关性。普通的异常检测可能无法捕捉到异常，这时可以考虑**多变量高斯分布**：$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$，其中$\Sigma$是协方差矩阵。

**异常检测算法：**
Choose features $x_i$ that you think might be indicative of anomalous examples.
Fit model $p(x)$ by setting:
	\- $\mu_j=\frac{1}{m}\sum_{i=1}^mx_j^{(i)}$
	\- $\Sigma=\frac{1}{m}\sum_{i=1}^m (x^{(i)}-\mu)^T(x^{(i)}-\mu)$
Given a new example x, compute:
	\- $p(x)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$
Flag 1 if $p(x)<\epsilon$.

原模型实际上是多变量高斯分布在$\Sigma$为对角矩阵时的特殊情形，其沿轴对称分布.

**原模型：**$\prod_{j=1}^n p(x_j;\mu_j,\sigma_j^2)$
\- 可以手动对原特征$x_1, x_2$进行组合得到新特征，用新特征进行训练
\- 计算代价小，对较大的特征数$n$适应能力强
\- 在训练样本数量$m$较小时也能运行

**多元高斯分布：**$p(x)=\frac{1}{(2\pi)^\frac{n}{2}|\Sigma|^\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$
\- 自动捕捉特征间关联
\- 计算代价高
\- 要求$m>n$，否则$\Sigma$可能不可逆($\Sigma$不可逆的概率很小，如果发生了通常是因为$m<n$或存在冗余特征)。通常在$m\gg n$时才会考虑多元高斯分布的异常检测.

原模型在实践中应用更加广泛。

***

## L16 - 推荐系统 Recommender Systems

### 1 问题描述

**推荐系统**在现实中的应用场景非常广泛，很多科技公司都致力于提高推荐系统的性能。另外，我们已经看到特征的选取在机器学习中的重要性，现在一些系统能够自动选取合适的特征，而推荐系统就是其中之一。

推荐系统问题的一个主要形式是：根据已有的用户信息，填补用户未知的信息。以电影评分为例，大量用户会为许多电影评分，我们需要根据已知信息，推测用户对其未看过的电影的评分。

**Notation:**
$n_u$：用户数量
$n_m$：电影数量
$r(i,j)$：当用户$i$给电影$j$评过分时为1
$y^{(i,j)}：$用户$i$给电影$j$的评分，在$r(i,j)=1$时有定义
$m^{(j)}$：用户$j$评价的电影数量

### 2 基于内容的推荐算法

**基于内容推荐算法 content based recommendation**假设每部电影已知其特征向量$x^{(i)}$(该算法中加上$x_0=1$)，希望为每个用户找到参数$\theta^{(j)}$，通过$(\theta^{(j)})^Tx^{(i)}$可以计算出用户$i$对电影$j$的评价.
为了找到$\theta^{(j)}$，需要确定优化目标。对于单个用户$j$，优化目标为：$\min_{\theta^{(j)}}\frac{1}{2m^{(j)}}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2m^{(j)}}\sum_{k=1}^n(\theta^{(j)}_k)^2$
整体形式和线性回归很像，第二项时正则项。式子可以乘以$2m^{(j)}$以简化常数。

推荐体统需要为每一个用户都进行优化，因而**优化目标**是：
$$
\min_{\theta^{(1)},\cdots,\theta^{(n_u)}}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta^{(j)}_k)^2
$$
使用梯度下降法进行优化时，循环中的式子为：
$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} $ (for $k=0$)
$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} + \lambda\theta_k^{(j)} \right) $ (for $k\neq 0$)
从这里也可看出基于内容推荐算法与线性回归的相似之处.

### 3 协同过滤

基于内容的推荐算法假设每部电影的特征都是已知的，但现实中确定每部电影的各个特征代价是很大的。如果用户在提供评分之外，也提供了自己对每种电影特征的偏好程度，即参数$\theta^{(j)}$，我们也可以通过类似的办法确定电影的参数$x^{(i)}$。
这种情况下，对于每部电影优化目标是：$\min_{x^{(i)}}\frac{1}{2}\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{k=1}^n(x^{(i)}_k)^2$
考虑整个系统，优化目标是：
$$
\min_{x^{(1)},\cdots,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x^{(i)}_k)^2
$$
可以发现，假设每个用户都给几部电影评了分，而每部电影都有几个用户评价过，这时可以在给定$\theta^{(1)},\cdots,\theta^{(n_u)}$的情况下学习$x^{(1)},\cdots,x^{(n_m)}$，也可以在给定后者的情况下学习前者。

**协同过滤 collaborative filtering**的思想是：先猜想$\theta$，通过$\theta$可以学习$x$，然后再学习$\theta$，如此循环往复，不断优化$\theta,x$。这个过程中每个用户提供评分信息都会提高推荐系统的准确性，因而谓之协同。在此基础上，**协同过滤算法**省去了$x$和$\theta$之间的来回迭代，用一个代价函数同时优化$\theta$和$x$：
$$
J(x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)})=\frac{1}{2}\sum_{(i,j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\\\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n(x^{(i)}_k)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n(\theta^{(j)}_k)^2
$$
这个代价函数合并了之前两个过程的优化目标，可以实现同时优化$\theta$和$x$，从而将这个算法变得和其他机器学习算法类似。

要注意的是，由于特征$x$需要学习得到，这里放弃$x_0=1$的做法，规定$x\in\mathbb{R^n}$

**协同过滤算法：**
Initialize $x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)}$ to small random values.
Minimize $J(x^{(1)},\cdots,x^{(n_m)},\theta^{(1)},\cdots,\theta^{(n_u)})$ with gradient descendent for every $j=1,\cdots,n_u$, $i=1,\cdots, n_m$:
	$x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})\theta_k^{(j)} + \lambda x_k^{(i)} \right)$
	$\theta_k^{(j)} := \theta^{(j)}_k -\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_k^{(i)} + \lambda\theta_k^{(j)} \right) $
For a user with parameters $\theta$ and a movie with (learned) features $x$, predict a star rating of $\theta^Tx$.

### 4 协同过滤的应用与实现

**向量化实现**：用矩阵$Y$记录用户评分，其中可能存在未定义数据。计算用户预估评分时，可用矩阵$X=\left[ \begin{matrix} (x^{(1)})^T\\(x^{(2)})^T\\\cdots\\(x^{(n_m)})^T \end{matrix} \right]$记录电影特征，矩阵$\Theta=\left[ \begin{matrix} (\theta^{(1)})^T\\(\theta^{(2)})^T\\\cdots\\(\theta^{(n_u)})^T \end{matrix} \right]$记录用户参数，$X\Theta^T$即为预估分数。

**相似推荐：**算法通常可以学习到一些非常关键的特征，但是这些特征对人们来说往往难以理解。在推荐相似电影时，可以根据$\|x^{(i)}-x^{(j)}\|$来计算出与用户当前看过的电影中最相似的几个。

**均值归一化：**如果一个用户从未评价过任何电影，训练出的$\theta$会是全零的，这样其预估的电影评分都会是0。如果这不是我们所希望的，可以采用均值归一化。先对$Y$每行有效数据求平均得到向量$\mu$，然后$Y$每列减去$\mu$进行训练。然后计算预估值的时候，推测用户$j$给电影$i$的评分是$(\theta^{(j)})^Tx^{(i)}+\mu_i$，这样没有评价记录的用户算出的预估评分即为平均值。

***

## L17 - 大规模机器学习 Large Scale Machine Learning

### 1 大数据集

很多应用场景中，数据量会达到上亿的级别。在很多模型中，这样的$m$会使得计算代价相当大。例如，梯度下降法中的$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$需要对$m$项求和。
可以考虑在$m$条数据中随机选择一部分进行训练，可通过绘制$J_{train}(\theta)$和$J_{cv}(\theta)$随着$m$变化的曲线来观察多大的$m$是可行的。
另外，也有一些方法可以用来解决过大的$m$带来的计算上的问题。

### 2 随机梯度下降法

许多机器学习算法中都有一个代价函数或优化目标，我们通常用梯度下降法进行优化。以线性回归为例，我们常用的算法被称作批量**梯度下降法 Batch Gradient Descendent**，因为每次循环都需要遍历数据集中全部$m$条数据：

**批量梯度下降法：**
$J_{train}(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})^2$
Repeat \{
	$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$
	(for every $j=0,\cdots,n$)
\}

在$m$达到上亿条时，每次梯度下降法走一步遍历所有数据的开销都非常大。为了大规模数据集上批量梯度下降法的问题，我们可以采用**随机梯度下降法 Stochastic gradient descendent：**

**随机梯度下降法：**
$cost(\theta,(x^{(i)},y^{(i)}))=\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2$ 
$J_{train}(\theta)=\frac{1}{m}\sum_{i=1}^mcost(\theta,(x^{(i)},y^{(i)}))$
Randomly shuffle (reorder) training examples.
Repeat \{
	for $i=1,\cdots,m$ \{
		$\theta_j:=\theta_j-\alpha(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$
		(for $j=0,\cdots,m$)
	\}
}

随机梯度下降法首先将数据随机打乱，然后在内循环中遍历每条数据。与批量梯度下降法不同的是，它在遍历每条数据时都会更新$\theta$。由于$m$很大，通常外循环进行1次就可以得到很好的假设函数了，一般外循环不超过10次。
从图像上看，批量梯度下降法每次循环都移动一步，移动形成的曲线通常平滑并直接指向最低点，最终在最低点收敛。随机梯度下降法由于每次用一条数据进行优化，因此移动有随机性，但是大体趋势还是向最低点靠近，最终在最低点附近的区域内活动。总体来看，随机梯度下降法的速度远高于批量梯度下降法，并且能得到一个很不错的假设函数。

**随机梯度下降法收敛：**批量梯度下降法中为了判断是否收敛，通常会绘制$J_{train}(\theta)$随着迭代次数的变化曲线，若最终趋于水平则已收敛。随机梯度下降法计算$J_{train}(\theta)$的代价过大，我们旨在内循环每次迭代时计算$cost(\theta,(x^{(i)},y^{(i)}))$，每进行一定次数迭代(如1000次)就绘制出这些$cost$的平均值，如果这条曲线趋于水平则收敛。另外，随机梯度下降法的曲线噪声可能比较大，可以增大迭代次数的间隔来求平均值、绘图。

**学习速率**$\alpha$：通常情况下$\alpha$用常数即可。随机梯度下降法最后会在最小值附近徘徊，但这影响不大。如果想要更精确的结果，也可以让$\alpha$逐渐变小，从而提高精度，例如使用$\alpha=\frac{constant_1}{iterationNum+constant_2}$.

### 3 Mini-Batch梯度下降法

批量梯度下降每次迭代处理$m$条数据，随机梯度下降每次处理1条数据。可以设想一种算法，每次处理$b$条数据，这就是**Mini-Batch梯度下降法**：

**Mini-Batch梯度下降法：**
Say $b=10, m=1000$
Repeat \{
	for $i=1,11,\cdots,991$ \{
		$\theta_j:=\theta_j-\alpha\frac{1}{10}\sum_{k=i}^{i+9}(h_\theta(x^{(k)})-y^{(k)})x^{(k)}_j$
		(for every $j=0,1,\cdots,n$)
	}
\}

Mini-Batch梯度下降法在合适的向量化计算下有时比随机梯度下降法更快.

### 4 在线学习

**在线学习 online learning**放弃了固定数据集的概念，在有连续数据流的情况下可以正常运行，在很多网站中会运用。另外，在线学习还有适应用户偏好变化的功能。如果不能保证不断的数据流，不建议使用在线学习。

假设某网站提供运输服务，特征$x$包含用户信息、出发地/目的地以及提供的价格，$y=1$表示用户购买了服务。为了训练参数$\theta$以预测用户的购买概率$p(y=1|x,\theta)$，可以用在线学习算法：

**在线学习：**
Repeat forever \{
	Get $(x,y)$ corresponding to user.
	Update $\theta$ : $\theta_j:=\theta_j-\alpha (h_\theta(x)-y)x_j$ (for $j=0,1,\cdots,n$)
\}

### 5 Map-reduce

一些规模庞大的机器学习问题用随机梯度下降法都难以解决，有时我们需要考虑用多台机器并行运算以提高计算速度，这就是**Map-reduce方法**。

假设$m=400$，需要计算$\theta_j:=\theta_j-\alpha\frac{1}{400}\sum_{i=1}^{400}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$，可以将求和的部分分到四台计算机上运行，例如第一台计算$temp_j^{(1)}=\sum_{i=1}^{100}(h_\theta(x^{(i)})-y^{(i)})x^{(i)}_j$。最后斯台计算机将结果传输给中心处理器，由它计算$\theta_j:=\theta_j-\alpha\frac{1}{400}(temp_j^{(1)}+temp_j^{(2)}+temp_j^{(3)}+temp_j^{(1=4)})$。如果不考虑网络延迟等因素，理论上计算速度会提高四倍。

需要大量求和的机器学习算法基本都可以用Map-reduce提速。另外，现在很多计算机的处理器都是多核的，也可以在多个核上应用Map-reduce方法。

***

**L18 - 机器学习应用: Photo OCR**
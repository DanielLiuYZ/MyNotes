# Machine Learning I

Author: 刘闵
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
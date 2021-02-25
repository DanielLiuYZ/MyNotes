# Python Notes

Author: 刘闵
Contact me:  liumin20010301@qq.com
Course: NJU AI Programming

[TOC]



## Chapter 1 - 数据类型

### 1 基本数据类型

class 'int'
	Python中整型与长整型不区分

class 'bool'
	True - 1, False  - 0 (注意T和F大写)

class 'float'
	ex: 3.22  9.8e3  -4.78e-2

class 'complex'
	虚部加上j，ex: 5+1j  2.4+5.6j

### 2 序列

#### 2.1 序列

序列是一种最基本的容器，主要包括字符串，列表，元组和range对象

##### 2.1.1 序列索引

sequence[index]，index可以为$-N\sim N-1$

##### 2.1.2 序列相关操作

**标准类型运算符：**
值比较： <  >  <=  >=  ==  !=
对象身份比较： is  is not
布尔运算： not  and  or

**序列类型运算符：**
切片：s[startidx : endidx]，idx缺省时默认为头、尾
			s[stridx : endidx : steps]
**重复：**sequence * copies，例如'apple' * 3为'appleappleapple', f = [0] * 20
连接：sequence1 + sequence2
判断成员：obj in sequence / obj not int sequence

**内建函数：**
转换内建函数：list()  str()  tuple()
其他常用内建函数：enumerate()  len()  reversed()  sorted()  max()  sum()  
							min()  zip()   //注意这些函数的参数以及返回值

#### 2.2 字符串

字符串可用单引号、双引号、三引号，其中三引号支持分行输入(' \\'为续行符)

字符串创建后**不可更改**其中字符内容

注意常用**转义字符**

**字符串常用方法：**与内建函数fun(str)不同的是，这里用str.fun()
str.center(11)  # 11个字符，中间n个是str
str.count(substr)  # 计算str中substr的个数
str.lower()  / str.upper()  # 大小写
str.find(subtr)  # 返回substr在str中首idx，可加两个参数表示考虑的字符串范围
str.join(tuple or list of string) #  用str连接tuple或list中的各个string
str.replace(astr, bstr) #  将str中的astr替换为bstr
str.split() # 根据' '分割str，返回list，或者用一个参数作为分割的字符

#### 2.3 列表

列表可包含不同类型的对象，可创建空列表。列表是可变的。

列表的**创建方法：**
aList = []
pList = [1, 'BA', 'The Boeing Company', 184.76]
cList = [x for x in range(1, 10, 2)]
dList = list('Python')

**列表的方法：**
list.append(ele) # 向list结尾添加一个元素
list.extend(alist) # 向list结尾添加一个序列中的所有元素
blist = alist.copy() # 浅拷贝，对容器元素仅复制地址
PS: blist = copy.deepcopy(alist) # 深拷贝
list.pop() # 默认弹出最后一个元素，可添加idx参数指定弹出元素
list.remove(ele) # 删除list中第一个ele
list.reverse() # 将list中元素倒序，reversed(list)会生成一份新的倒序的list
list.sort() # 有两个可选参数list.sort(key = None, reverse = False)
注意字符串和元组都没有reverse, sort方法，因为它们是不可变的，但是它们可以使用序列的内建函数reversed, sorted

PS: **array**模块
Python中list和tuple等数据结构可以表示数组，也可以用array模块通过array函数创建数组，如array.array('B', range(5))。array要求成员是同一类型，且在创建时就确定了类型。提供了append, insert和read等方法

#### 2.4 元组

元组是不可变的，但其可变元素可变。

元组**应用场合：**
· 在映射类型中当作键使用
· 函数的特殊类型参数
· 未明确定义的一组对象

元组**操作：**序列通用操作

#### 2.5 RANGE对象

用range()函数生成range对象，执行时一边计算一边产生值（类似一个生成器），生成一个不可变的整数序列。

range(start, end, step = 1)
range(start, end)
range(end)  # 注意区间为[start, end)

通常将range函数的返回值处理成list，如list(range(0, 10, 2))

### 3 字典

字典是一种映射类型，包含key-value对，其中键是唯一的: 数字、字符串、元组。字典中的元素是无序的。

#### 3.1 创建字典

**直接创建：**
aInfo = {'Mayue': 3000, 'Lilin': 4500, 'Wuyun': 8000}
aInfo[key] = val

**用dict()函数创建：**
aInfo = dict(bInfo) # 这里bInfo可以是一个tuple或list，每个元素也可以是tuple或list; 此外，bInfo还可以是几个用逗号隔开的赋值语句，例如Mayue = 3000，这里Mayue会自动被处理为字符串

**用方法fromkeys(seq [, value])创建：**
aInfo = {}.fromkeys(('Mayue', 'Lilin', 'Wuyun'), 3000)
得到{'Mayue': 3000, 'Lilin': 3000, 'Wuyun': 3000}

#### 3.2 生成字典

**用已有的两个列表aList, bList生成字典：**
aInfo = dict(zip(aList, bList))

#### 3.3 字典的基本操作

**键值查找：** aInfo['Lilin']

**字典更新：** aInfo['Lilin'] = 10000

**添加元素：** aInfo['Liuxi'] = 12000

**成员判断：** 'Liuyun' **in** aInfo

**删除元素：** **del** aInfo['Lilin']

#### 3.4 字典的内建函数

dict()  len()  hash()

#### 3.5 字典方法

aInfo.keys() # 返回dict_keys类型，包含所有keys
aInfo.values() # 返回dict_values类型，包含所有values
aInfo.items() # 返回dict_items类型，包含所有key-value tuple
aInfo.get(key) # 返回value，若不存在返回None，若get(key, value)，则在不存在的情况下返回value参数，这种方法相比于直接的键值查找，不会因为找不到而报错
aInfo.setdefault(key, val) # 若key存在或key后加上一个None参数，效果和get一样，若不存在且参数不为None，则在字典中添加key-val对
bInfo = aInfo.copy() # 拷贝字典
aInfo.pop(key) # 弹出指定的key-val
aInfo.clear() # 清空字典
aInfo.update(bInfo) # 用字典b更新字典a，若a中不存在相应key，添加key-val，否则值更新相应的val

### 4 集合

集合是一个**无序**、**不重复**的元素的组合，可用来**去重**。
包括可变集合(set)和不可变集合(frozenset)两种，也用大括号表示

#### 4.1 创建集合

aSet = set(seq)
fSet = frozenset(seq)

#### 4.2 集合的基本操作

| 数学符号    | Python符号 |
| ----------- | ---------- |
| $\in$       | in         |
| $\not\in$   | not in     |
| $=$         | ==         |
| $\neq$      | !=         |
| $\subset$   | <          |
| $\subseteq$ | <=         |
| $\supset$   | >          |
| $\supseteq$ | >=         |
| $\cap$      | &          |
| $\cup$      | \|         |
| $-$ 或 \    | -          |
| $\Delta$    | ^          |

运算符可复合：&=, |=, -=, ^=

#### 4.3 集合方法

**面向所有集合：**

aSet.issubset(t)
aSet.issuperset(t)
aSet.union(t)
aSet.intersection(t)
aSet.difference(t)
aSet.symmetric_difference(t)
aSet.copy()

**面向可变集合：**
aSet.update(t)
aSet.intersection_update(t)
aSet.difference_update(t)
aSet.symmetric_difference_update(t)
aSet.add(obj)
aSet.remove(obj) # 移除不存在的元素时会出错
aSet.discard(obj) # 移除不存在的元素时不会出错
aSet.pop() # 随机移除一个元素，返回移除的元素
aSet.clear()

****

## Chapter 2 - 程序控制结构

### 1 顺序结构

#### 1.1 赋值语句

普通赋值：r = 2
增量赋值：m /= 5
链式赋值：c = b = a
多重赋值：p, r = 3, 5    x, y = y, x (本质是元组打包与序列解包)

#### 1.2 输入/输出

**输入函数：**x = input(['输入提示'])，返回类型是str
常用函数搭配：
x, y = input('Enter 2 str: ').split() # split(',')
lst = list(eval(input('Enter a list: '))) # eval函数将str作为表达式执行并返回结果

**输出函数：**print(obj1, obj2, ... objn, sep = ' ', end = '\n')
基本格式化输出：'%d ...' % obj1, obj2, ... , objn
格式化模板：'格式化模板'.format(obj1, obj2, ... objn)
{参数位置: \[对齐说明符\]\[符号说明符\]\[最小宽度说明符\]\[.精度说明符\]\[类型说明符\]}
ex: print("Age: {0:<5d}, Height: {1:5.2f}".format(age, height))
Age: 21   , Height: 1.76

**输出符号：**
类型说明符：b - 二进制； o - 八进制； x - 十六进制； c - 字符； 
						 d - 十进制数； f - 定点数； e - 指数记法
[+]m.nf：'+'表示大于0带正号，保留n位小数，整个输出占至少m列
0>5d：右对齐，不足用0填充，输出项宽度最小为5
<：左对齐，默认用空格填充右边，<前后可添加填充字符和宽度
^：居中对齐
{{}}：输出一个{}

### 2 选择结构

#### 2.1 if - elif - else

注意Python通过缩进判断语句序列是否在同一代码块

```python
if expr1:
    block1
elif expr2:
    block2
else:
    block3
```

#### 2.2 嵌套的if语句

```python
if expr1:
    if expr2:
        blk1
    else:
        blk2
else:
    if expr3:
        blk3
    else:
        blk4
```

#### 2.3 三元运算符

形式： x if C else y
ex： t = x if x >= y else y  # t = max(x, y)

### 3 循环结构

#### 3.1 while循环

```python
while expr:
	语句序列(循环体)
```

#### 3.2 for循环

```python
for 变量 in 可迭代对象:
    语句序列
    
# Examples:
# 序列迭代
for i in range(1, 5):
	print(i * i)
# 序列索引迭代
s = ['I', 'love', 'Python']
for i in range(len(s))
	print(s[i], end = ' ')
# 迭代器迭代
courses = ['Maths', 'English', 'Python']
scores = [88, 92, 95]
for c,s in zip(courses, scores):
    print(('{0} - {1:d}').format(c, s))
# 其他迭代
d_stock = {'AXP':'78.51', 'BA':'184.76', 'CAT':'96.39'}
for k,v in d_stock.items():
    print('{0:>3}: {1}'.format(k, v))
for k in d_stock.keys():
    print(k, d_stock[k])
    
```

**可迭代对象：**可以按次序迭代的对象，包括序列、迭代器(iterator)如enumerate()产生的对象，以及其他可以迭代的对象如字典的键和文件的行等
(序列、序列索引、字典的键、文件的行、迭代器)

**PS. 迭代器：** 迭代器是访问集合元素的一种方式，可以记住遍历的位置，只能往前移动，不可后退。常用函数有iter()和next()。很多函数的返回值是迭代器。

#### 3.3 break, continue

break语句终止当前循环，转而执行循环之后的语句

continue语句跳过循环体内continue后的语句，开始新的一轮循环

#### 3.4 列表解析

一种特殊的循环，通过for语句结合if语句，利用其他列表动态生成新列表，特殊的轻量级循环。其中多个for语句相当于for结构的嵌套使用：

[表达式  for 表达式1 in 序列1
				for 表达式2 in 序列2
				...
				for 表达式N in 序列N
				if 条件]

ex: [x\**2 for x in range(10) if x\*\*2 < 50]
	  [(x+1, y+1) for x in range(2) for y in range(2)]

****

## Chapter 3 - 函数

函数是一个独立代码块，为每个独立代码块定义一个名字以及能与其他独立代码块通信的接口即为函数。

函数分类：

1.内建函数 Built-in Functions
2.标准库函数
	库是一组具有相关功能的模块的集合，需要先导入模块再使用函数
	from math import *  #pi
	from math import pi	#pi
	import math	#math.pi
3.第三方库
4.用户自定义函数

### 1 常用Python内建函数

**数值型内建函数：**
abs()	bool()	oct()
round()	int()	hex()
divmod()	ord()	pow()
float()	chr()	complex()

**部分实用函数：**
dir()	input()	help()
open()	len()	range()

### 2 常用Python标准库函数

常见标准库模块包括os, shutil, re, math, ramdom, sys, datetime等

**os模块：** 处理文件及目录
os.getcwd() # 返回当前工作目录
os.chdir(path) # 切换当前工作目录
os.rename('current.txt', 'new.txt')
os.remove('new.txt')
os.mkdir('d:\\\\temp\\\\tempdir')
os.rmdir('d:\\\\temp\\\\tempdir')

**sys模块：** 标准输入和输出属性
sys.stdin, sys.stdout

**random模块：**
random.seed(100)
random.random()
random.choice(lst) # 设定随机生成结果的范围
random.randint(a, b) # 这里的randint生成[a, b]范围内的一个整数，注意是闭区间
random.randrange([start,] stop[, step]) # 设定随机生成结果的范围，不包含stop
random.uniform(a, b) # 生成[a, b)的任意实数
random.sample(seq, num) # 生成seq范围内num个随机结果，返回list
random.shuffle(seq) # 将seq元素随机排序

**datetime模块：**
datatime.date.today() # 日期
datetime.datetime.now() # 日期 时间
time = datetime.time(23, 20, 35) # 返回时间
time = datetime.data.today() + datetime.timedelta(days = 3) # 三天后
time = datatime.datatime.now().strftime('format') # 将时间转化为指定的字符格式 %a(A)-星期, %b(B)-本地月份, %d-日期, %y(Y)-年份, %H(l)小时, %M-分钟
ts = dt.timestamp() # 时间戳
tm = datetime.fromtimestamp(ts) # 从时间戳获取时间

### 3 自定义函数

#### 3.1 自定义函数的创建

```python
def fun_name([parameters]):
    ['''DocStrings''']
    function body
```

#### 3.2 函数的返回

```python
def square(x, y):
	'''
	calculate sum and div
	'''
    return x + y, x - y
```

返回格式：return expr1, expr2, ... , exprn

#### 3.3 函数的调用

```python
x, y = square(3, 5) # use the example in 3.2
```

#### 3.4 函数的参数

**1. 位置参数：**

```python
printGrade('Danny', '35', 'A+')
```

**2. 关键词参数：**

```python
printGrade(name='Danny', stuID='35', grade='A+')
```

让调用者通过使用参数名区分参数，允许改变参数列表中的参数顺序，调用时每个参数的含义更清晰。注意使用关键词参数时尽量统一，若和位置参数混用但参数位置顺序与定义不一致会引发问题。

**3. 默认参数：**

```python
def printGrade(name, stuID, grade = 'A'):
	...
```

默认参数必须放在所有非默认参数后

**4-a. 可变长参数-可变长位置参数：**

```python
def fun(args1, *tupleArgs):
    ...

fun(args1, args2, args3)
```

形参前*是可变长位置参数的标记，收集其余的位置参数放在一个元组中。调用时若将一个tuple作为参数加上\*，则等价于传递tuple元素作为参数

**4-b. 可变长参数-可变长关键字参数：**

```python
def fun(**dictArgs):
    ...

fun(a = args1, b = args2)
```

\*\*表示可变长的关键字参数，允许传入多个(0个)含参数名的参数，这些参数在函数内自动组装成一个字典。同样地，一个字典作为参数加上\*\*，则等价于传递字典内的key = value作为参数

#### 3.5 递归函数

```python
def f():
    ''' infinite recursion '''
    f()
```

**递归深度的设定：**
sys.getrecursionlimit() # 查看递归深度
sys.setrecursionlimit(2000) # 手工修改默认值

#### 3.6 LAMBDA函数和函数式编程

lambda函数又称匿名函数，即没有具体的函数名。lambda函数的目的时让用户快速定义单行函数，简化用户使用函数的过程。

```python
my_add = lambda x, y : x + y
my_add(3, 5)
```

**函数式编程：**由3个基本函数和1个算子构成
基本函数：
map(fun, seq) # 对seq中的item依次执行fun(item)，返回结果为fun(item)列表
filter(fun, seq) # 对seq中的item依次执行fun(item)，将返回True的item作为一个seq返回
reduce(fun, seq [, initial]) # fun只能接受两个参数，将seq中第1和第2个item作为参数，返回结果与第3个item作为参数，依次迭代，得到结果
算子：lambda

```python
reduce(lambda x, y: x + y, [1, 2, 3, 4, 5]) 
# return 15
list(filter(lambda x: x % 2 == 0, [3, 2, 5, 8, 1])) 
# return [2, 4, 6]
list(map(map(lambda x: x * 2, [3, 2, 5, 8, 1]))) 
# return [6, 4, 10, 16, 2]
```

#### 3.7 变量作用域

**作用域**会生成一个命名空间
**命名空间**是从名称(标识符)到对象的映射
不同命名空间中的名字之间没有关系

局部变量(包括形参)与全局变量同名时，局部变量屏蔽全局变量。使用关键字**global**声明将使用全局变量(ex: global x)

****

## Chapter 4 - 面向对象程序设计

### 1 面向对象程序设计基本概念

**对象：**由数据及能对其实施的操作所构成的封装体
**类：**类描述了对象的特征(数据和操作)

面向对象程序设计的**基本特征：**
· **抽象**与**封装**
	-抽象是指对现实世界问题和实体的本质表现；问题分解成数据和数据上的操作
	-封装是将程序具体实现的细节进行隐藏的一种机制
· **多态**和**绑定**
	-多态是指一个事物有多种不同的解释，根据传递参数的不同执行不同的函数或操作不同的代码
	-绑定是指在具体某次使用多态元素时确定使用的时哪一种形式
· **继承**
	-新创建的类的一些特征(包括属性和方法)可以从其他已有的类获得

### 2 类与对象

#### 2.1 类的定义和方法

```python
class ClassName:
    '''DocStrings'''
    ClassBody # some property and methods
    
class ClassName(object): # for inheritance 
    '''DocStrings''' # object: source of all classes
    ClassBody
```

#### 2.2 实例

实例化形式：**变量 = 类名(<参数>)**
创建实例后，可以使用实例调用方法
类方法的第一个参数总是**self**，指向实例本身，Python自动将对象作为第一个参数传入方法中

#### 2.3 \_\_init\_\_()方法

\_\_init\_\_()方法永远在对象创建完成后被Python自动调用，是对象被创建后Python自动调用的第一个算法。和其它方法一样，本身作为self参数传递

```python
class Dog(object):
    '''define Dog Class'''
    def __init__(self, name):
        self.name = name
    def greet(self):
        print("Hi, I am %s" % self.name)
        
if __name__ == "__main__":
    dog = Dog("Paul")
    dog.greet()
```

#### 2.4 实例属性

实例属性**创建时间**：定义类时或实例创建之后
所有实例属性保存在名为\_\_dict\_\_的内嵌属性里

```python
class Dog(object):
    '''define Dog class'''
    counter = 0 # 定义类时创建的属性，可追踪已创建实例个数
    def __init__(self, name):
        self.name = name # 具体对象的属性
        Dog.counter += 1 # 类中所有对象属性
        self.greet()
    def greet(self)
    	print("Hi, I am {:s}, my number is {:d}".format(self.name,	Dog.counter))
        
```

### 3 类的继承和方法重写

#### 3.1 子类的定义

```python
class SubClassName(ParentClass1[, ParentClass2, ...]):
    '''DocStrings'''
    ClassBody
```

可分为单继承与多继承

#### 3.2 子类重写

```python
class p_father(object):
    def pnt(self, a, b):
    	self.a = a
        self.b = b
        print(self.a + self.b)
        
class p_child(p_father):
    def pnt(self, a, b):
        self.a = a
        self.b = b
        print(self.a + self.b)
        print(self.a * self.b)
    # use super() to override:
    def pnt(self, a, b)
    	super().pnt(a, b)
        print(self.a, self.b)
```

当子类中方法重写内容包含父类时，可用super()方法

**注意：**
python中不支持其它语言中常见的函数重载(即同名不同参的函数)。这是因为Python可以接受任何类型的参数，且接受可变长参数，因而无需函数重载即可解决可变参数类型和可变参数个数的问题

#### 3.3 访问控制

类中 \_\_var (前面两个下划线)命名的属性外界无法访问。\_\_var\_\_则又是可以在外界访问的，不过普通属性不会使用这种命名方式。
\_\_var属性会被\_classname__var替换，将防止父类与子类中出现的重名冲突

### 4 常用类和实例相关内建函数

issubclass(classa, classb)
isinstance(obj, classn)
super(type, obj = None)
hasattr(obj, attr)
setattr(obj, attr, val)
getattr(obj, attr[, default])
delattr(obj, attr)
vars(obj = None)
dir(obj = None)

****

## Chapter 5 - 异常处理

### 1 Python中的异常

程序设计错误：语法错误、运行时错误、逻辑错误
异常：用**异常对象**(exception object)表示异常情况

**异常类：**可通过dir(__builtins__)查看

| 类名              | 描述                          |
| ----------------- | ----------------------------- |
| BaseException     | 所有异常的基类                |
| Exception         | 常规异常的基类                |
| AttributeError    | 对象不存在此属性              |
| IndexError        | 序列中无此索引                |
| IOError           | 输入/输出操作失败             |
| KeyboardInterrupt | 用户中断执行(Ctrl+C)          |
| KeyError          | 映射中不存在此键              |
| NameError         | 找不到名字(变量)              |
| SyntaxError       | Python语法错误                |
| TypeError         | 对类型无效的操作              |
| ValueError        | 传入无效的参数                |
| ZeroDivisionError | 除(或取模)运算的第二个参数为0 |

### 2 异常处理

#### 2.1 异常捕捉 try-except语句

try-except中可包含多个**except子句**，一个except块可以**捕捉多个异常**。如果没有发生异常且存在**else子句**，则执行else块。不论是否发生异常，有**finally子句**时都需要执行finally块(注意：发生异常时，try中剩余代码不会执行)：

```python
try:
    blockToExamine
except Exception1:
    blockForException1
except Exception2:
    blockForException2
except (Exception3, Exception4): # multiple exptions
    blockForException3
else:
    blockForNoException
finally:
    blockMustRun
```

**空except子句**可以捕捉所有类型的异常且必须放在所有except后：

```python
try:
    blockToExamine
except:   # empty except for all exceptions
    blockForException
```

**as子句**可以帮助获取错误原因信息(string)，用法如下：

```python
try:
    blockToExamine
except ExceptionName as err:
    blockForException
    print(err) #err is a str like 'division by zero'
```

#### 2.2 上下文管理器和with语句

**上下文管理器**定义和控制代码块执行前的**准备动作**及执行后的**收尾动作**(\_\_enter\_\_()方法和\_\_exit()\_\_()方法)，通过with语句在支持上下文管理协议的对象(如文件对象)上方便地进行使用。
**with语句执行顺序：**
当with语句遇到上下文管理器，就会在执行语句体之前，先执行上下文管理器的\_\_enter\_\_()方法，然后再执行语句体，执行完语句体后，最后执行\_\_exit\_\_()方法。
**\_\_exit\_\_ ()：**
出现异常时，如果\_\_exit\_\_ ()返回 False（默认不写返回值时，即为False），则会重新抛出异常，让with 之外的语句逻辑来处理异常，这也是通用做法；如果返回 True，则忽略异常，不再对异常进行处理。
**\_\_enter\_\_()：**
调用上下文管理器的\_\_enter\_\_()方法时，如果使用了 as 子句，则将\_\_enter\_\_()方法的**返回值赋值给 as** 子句中的目标

```python
with 上下文管理表达式 as 变量:
    语句序列
    
# method to deal with file exception
try:
    with open(r'path') as fp:
        blockForFile
except IOError as err:
    printI(err)
```

如果不采用with语句，该异常处理部分还需要添加finally语句执行close()。with语句简化代码的同时也提高了安全性。

#### 2.3 raise语句

```python
# math库中sqrt()函数的实现模拟
def sqrt(x):
    if not isinstance(x, (int, float)):
        raise TypeError('a float is required')
    elif x < 0:
        raise ValueError('math domain error')
    block # 计算平方根的功能模块
```

****

## Chapter 6 - 科学计算与数据分析

### 1 软件生态系统SciPy

**SciPy**是基于Python的开源软件生态系统，主要为数学、科学和工程服务

**Numpy：**
高性能科学计算和数据分析的基础包
强大的ndarray对象
精巧的函数和ufunc函数
适合线性代数和随机数处理等科学计算

**SciPy library：**
基于NumPy，是科学计算核心库
有效计算numpy矩阵，让NumPy和SciPy library协同工作
致力于科学计算中常见问题的各个工具箱，其不同子模块有不同的应用，如插值、积分、优化和图像处理等

**Matplotilib：**
基于NumPy
二维绘图库，简单快速地生成曲线图、直方图和散点图等形式的图
常用的pyplot是一个简单提供类似MATLAB接口的模块

**SymPy：**
数学符号Python计算库，全功能地计算机代数系统，多项式求值、求极限、解方程、求积分、微分方程、级数展开等计算问题
代码简洁、易于理解和扩展

**pandas：**
基于SciPy library和NumPy
高效的Series和DataFrame数据结构
强大的可扩展数据操作与分析的Python库
高效处理大数据集的切片等功能
提供优化库功能读写多种文件格式，如CSV, HDF5

### 2 NumPy与科学计算

#### 2.1 ndarray的基本特性

**ndarray：**
-ndarray是N维数组，是NumPy中的基本数据结构
-所有元素是同一种类型
-别名为array
-利于节省内存和CPU计算时间
-有丰富的函数

**基本概念：**
维度(dimensions)称为轴(axes)，轴的个数称为秩(rank)
基本属性：
ndarray.ndim # 秩
ndarray.shape # 维度
ndarray.size # 元素总个数
ndarray.dtype # 元素类型
ndarray.itemsize # 元素字节大小

#### 2.2 ndarray的创建

ndarray创建函数：
arange	array	copy	empty	empty_like	eye
fromfile	fromfunction	full	identity	linspace	logspace
mgrid	ogrid	ones	ones_like	r	zeros	zero_like ...

```python
import numpy as np
# array()
np.array([(1,2,3),(4,5,6)], dtype = float)
np.array([1,2,3])
# zeros() ones() full()
np.zeros([2,3]) # 2*3 全1 浮点
np.ones([2,3]) 	# 2*3 全0 浮点
np.full([2,3], np.pi) # 2*3 全pi 浮点
# zeros_like()
np.zeros_like(aArray) # 返回和aArray的shape全同的数组
np.ones_like(aArray)
np.full_like(aArray)
# identity()
np.identity(3) # 三阶单位阵
np.eye(3) # 三阶单位阵，eye可通过其他参数来设置成非方阵
np.diag(aArray) # 参数为一维数组，将其扩展为对角矩阵
				# 参数为二维矩阵，返回一维的对角元素
# arange() linspace()
np.arange([start,]stop,[step,]dtype=None) # 等差数组,
						#不含stop，默认从0，步长为1
np.linspace(start,stop,num=50,endpoint=True, retstep=False,dtype=None) # 等差数组
# random() rand() normal() randn() uniform() empty()
np.random.random(size) # [0,1), size可为int或tuple
np.random.rand(d0,d1,...,dn) # 指定shape[0,1)均分
np.random.normal(loc, scale, size) # 正态分布
np.random.randn(d0,d1,...,dn) # 指定shape标准正态分布
np.random.uniform(low=0,high=1,size=None) # [l,h]均匀分布
np.empty(shape,dtype=float,order='C') #不赋值, 慎用
# fromfunction()
np.fromfunction(function, shape, dtype=float)
# function参数的个数和shape的rank一样，shape每个位置的坐标即为function的参数
# choice()
np.random.choice(a, size=None, replace=True)
# 从1维数组a中随机选取元素返回size大小的结果，replace=True时表示可以重复选取某个元素
```

Example: 对一个二维数组，从中无放回地采样出k行数据

```python
A = np.random.rand(6, 3)
mask = np.random.choice(np.arange(A.shape[0]), k, replace = False)
Sample = A[mask]
```

#### 2.3 ndarray的操作与运算

##### 2.3.1 ndarray操作

**切片：**
aArray = np.array([(1, 2, 3), (4, 5, 6)])
aArray[1] # [4 5 6]
aArray[0:2] # [[1 2 3] [4 5 6]]
aArray[:, [0, 1]] # [[1 2] [4 5]]
aArray[1, 0:2] # [4 5]
注意用a:b表示[a, b)

**布尔索引：**
aArray = np.arange(1, 101)
aArray[aArray <= 50]
aArray[(aArray % 2 == 0) & (aArray > 50)]
aArray[(aArray % 2 == 0)] = -1
np.where(aArray % 2 == 0, -1, aArray) # 和上一条等价，where(cond[, x, y])若参数只有condition，返回True的位置索引元组，否则True返回x，False返回y

**改变数组形状：**
aArray = np.array([(1, 2, 3), (4, 5, 6)])
aArray.shape # 返回(2, 3)
aArray.reshape(3, 2) # 返回3行2列的array([[1,2],[3,4],[5,6]])，本身不变
aArray.resize(3, 2) # aArray本身会变为3行2列
aArray.ravel() # 数组展平，会改变原数据
aArray.flatten() # 数组展平，不改变原数据
以上函数也可写成func(aArray, param...)的形式

**数组组合：**
np.hstack((aArray, bArray, ...)) # 水平方向，注意只有一个参数
np.vstack((aArray, bArray, ...)) # 垂直方向

**数组分割：**
np.hsplit(aArray, n) # 沿axis=1分割，均分为n份(必可均分)，原值不变
np.vsplit(aArray, n) # 沿axis=0分割
np.split(aArray, n, axis = 1)
np.split(aArray, n, axis = 0)

##### 2.3.2 ndarray运算

**基本运算：**
利用基本运算符。较小的数组会光波导较大的数组的大小使其形状兼容(需要满足广播功能的条件). 
ex:
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3], [4, 5, 6]])
a + b # array([[2, 4, 6], [5, 7, 9]])

**统计：**
利用基本数组统计方法
aArray.sum() # 求和，若有**参数axis = 0**，则表示沿着0-axis分别求和
aArray.min() # 最小值
aArray.argmin() # 最小值的下标
aArray.max() # 最大值
aArray.argmax() # 最大值的下标
aArray.mean() # 均值
aArray.var() # 方差
aArray.std() # 标准差
aArray.sort(axis=-1, kind=None, order=None) # kind:算法，order:排序依据的key名，会改变原数据
aArray.argsort() # 返回排序后的下标
aArray.cumsum() # 依次累加，若不指定axis则视作一维累加
aArray.cumprod() # 依次累乘

##### 2.3.3 ufunc函数：

ufunc是一种能对数组的每个元素进行操作的函数。NumPy内置的很多ufunc函数都是在C语言级别实现的，计算速度非常快，数据量大时有很大的优势。
**Math operations:** add, subtract, mod, power, log, ...
**Trigonometric functions:** sin, cos, tan, tanh, ...
**Bit_twidlling funtions:** bitwise_and, invert, ...
**Comparison functions:** greater, less, equal, logical_and, ...
**Floating functions:** isinf, fabs, modf, floor, ...

**将标量函数转换为ufunc函数：**
np.frompyfunc(func, nin, nout)
\# nin：参数个数，int类型
\# nout：函数返回对象个数，int类型

```python
def f(x):
    return 2 * x

double_array = np.frompyfunc(f, 1, 1)
print(double_array(np.arange(1,1000)))
```

np.vectorize(func)
\# 将 f: a -> b 转化为 f: a[] -> b[]，返回函数

#### 2.4 线性代数应用

Scipy中的linalg模块可用于线性代数

np.matrix(seq) # 转化为矩阵，array也可用于运算
np.dot(m1, m2) # 矩阵内积
np.linalg.det(m) # 行列式
np.linalg.inv(m) # 逆矩阵
np.linalg.solve(A, b) # 多元一次方程组求根
np.linalg.eig # 求特征值和特征向量，返回一个特征值array和特征向量(按列对应排放)array组成的tuple

### 3 pandas与数据分析

#### 3.1 Series

Series是类似一维数组的对象，由数据和索引组成(有序字典，称**变长字典**)

**创建Series：**
import pandas as pd
aSer = pd.Series([1, 0.0, 'a']) # 索引默认0, 1, 2...
aSer = pd.Series(['apple','peach','lemon'], index=[1,2,3]) # 常自定义index
aSer.index # 返回index
aSer.values # 返回value

**Series的基本运算：**
基本运算：对value进行运算
aSer[1:2] # 基于位置切片
aSer['a':'b] # 基于索引切片

#### 3.2 DataFrame

##### 3.2.1 DataFrame的基本特性

DataFrame是一个表格型的数据结构(称数据框)，含有一组有序的列(类似于index)，大致可以看成共享同一个index的Series集合

ex:
			name		pay
0		Mayue        3000
1          Liulin       4500
2        Wuyun       8000

##### 3.2.2 创建DataFrame & 索引和值

aDF = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
\# data为输入数据
\# index为行索引，默认0, 1, ... 可通过aDF.index查看/修改
\# columns为列索引，默认0, 1, ... 可通过aDF.columns查看/修改
\# 可通过aDF.values得到值，结果为二维数组，每个元素是一行数据

ex:

```python
data = {'name':['Mayue','Lilin','Wuyun'], 'pay':[3000,4500,8000]}
aDF = pd.DataFrame(data)

data = np.array([('Mayue',3000),('Lilin',4500),('Wuyun',8000)])
bDF = pd.DataFrame(data, index=range(0,3), columns=['name','pay'])
```

##### 3.2.3 DataFrame操作

**添加列/行：**
aDF['tax'] = [0.05, 0.05, 0.1] # 添加列
aDF.loc[5] = {'name':'Liuxi', 'pay':5000, 'tax':0.05} # 添加行
aDF.append(bDF) # 添加行
aDF = pd.concat([aDF, bDF]) # 添加行

**删除列/行：**
aDF.drop(labels=None, axis=0, index=None, columns=None)
\# 删除，labels表示标签，axis=0删除行，axis=1删除列
\# index=lb等价于labels=lb；columns=lb等价于columns=lb, axis=1

**修改：**
aDF[lb] = data # 修改列
aDF.loc[lb] = data # 修改行
aDF[lb] = aDF[lb].astype(typename) # 修改列元素类型

**交换DataFrame元素：**
aDF.reindex(ind, axis = 0) # 根据ind的顺序重新排列行
aDF.reindex(col, axis = 1) # 根据col的顺序重新排列列

**数据选择：**
**a. 选择行**
aDF.['a' : 'c'] # 索引
aDF[0:3] # 切片
aDF.head(3) # 专门的方法
**b. 选择列**
aDF[listOfColName] # 列名，不可用a:b
aDF.colName
**c. 选择区域**
aDF.loc[la:lb, lc:ld] # 标签，a:b等都可用list替换
aDF.iloc[a:b, c:d] # 索引，注意标签用闭区间，索引半开半闭
**d. 选择单个值**
loc或iloc
aDF.at[la, lb] # 标签
aDF.iat[a, b] # 索引
**e. 条件筛选**
aDF[condition]
ex: aDF[(df.index >= 'b') & (df.index <= 'd' & (df.maths >= 90))]
ex: aDF[df.name.str.contains('Black')]

#### 3.3 基于Series和DataFrame的数据统计和分析

**简单统计与筛选：**
aDF.mean() # 对每列求均值
aDF.colName.mean() # 对某列求均值
aDF[condition].mean() # 对条件筛选出的数据求均值
aDF.max(axis=0) # 找出每列最大值，axis=1则找出每行最大值

**排序：**
aDF.sort_values(by = colName)

**分组：**
aDF.groupby(by = None) # 根据某列分组，返回groupby object
aDF.groupby(col).apply(len) # apply方法接受参数func，很常用
ex:

```python
df.groupby(month)['open', 'close'].mean()
df.groupby('grade').name.count()
```

**合并：**
append: 加行到DataFrame
concat: 连接pandas对象
join: SQL类型的连接

### 4 Matplotlib与可视化

Matplotlib是著名Python绘图库，主要用于二维绘图，画图质量高，方便快捷。其他绘图模块包括seaborn, pyechars等

#### 4.1 Matplotlib绘图基本方法

import matplotlib.pyplot as plt

**折线图：**
plt.plot([x,] y, [fmt,] [x2,] y2, [fmt2], ..., ) # x默认0,1,...,N-1

```python
t = np.arange(0., 4., 0.1)
plt.plot(t, t, t, t+2, t, t**2) # multiple
#plt.plot(t, t)
#plt.plot(t, t+2)
#plt.plot(t, t**2)
plt.show()
```

**散点图：**
plt.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, ...)

**条形图：**
plt.bar(x, height, width=0.8, bottom=None, ...)

**直方图：**反映数据分布
plt.hist( ... )

**箱型图：**
plt.boxplot(...)

#### 4.2 Matplotlib图形属性控制

Matplpotlib可以控制的默认属性：
文字和字体属性，坐标轴和网格属性，子图(axes)，子区(subplots)，
色彩和样式，线宽，每英寸点数，图像大小，...

##### 4.2.1 色彩和样式

| 符号 | 颜色         | 线型   | 描述         | 标记 | 描述            |
| ---- | ------------ | ------ | ------------ | ---- | --------------- |
| b    | blue         | '-'    | solid        | "o"  | circle          |
| g    | green        | '--'   | dashed       | "v"  | triangle_down   |
| r    | red          | '-.'   | dash_dot     | "s"  | square          |
| c    | cyan 青      | ':'    | dotted       | "p"  | pentagon 五边形 |
| m    | magenta 紫红 | 'None' | draw nothing | "*"  | star            |
| y    | yellow       | ' '    | draw nothing | "h"  | hexagon         |
| k    | black        | ''     | draw nothing | "+"  | plus            |
| w    | white        |        |              | "D"  | diamond         |

Example:

```python
plt.figure(figsize=(8,6), dpi=100)
t = np.arange(0., 4., 0.1)
plt.plot(t,t,color='r',linestyle='-',linewidth=3,label='Line1')
plt.plot(t,t+2,color='g',linestyle='',marker='*',linewidth=3,label='Line2')
plt.plot(t,t**2,color='b',linestyle='',marker='+',linewidth=3,label='Line3')
plt.legend(loc='upper left')
plt.show()
```

**文字：**
plt.title('Plot Example') # 标题
plt.xlabel('X label') # 横轴
plt.ylabel('Y label') # 纵轴

**绘制子图：**在一个图的多个区域分别绘图
Matplotlib中绘图子当前图形和当前坐标系中，默认在编号1的figure中绘图，可以通过subplot()函数和axes()函数绘制子图
plt.subplot(211) # 上下分，当前：上
plt.subplot(212) # 上下分，当前：下
plt.subplot(121) # 左右分，当前：左
plt.subplot(122) # 左右分，当前：右
plt.subplot(221) # 分四块，当前：左上
plt.subplot(224) # 分四块，当前：右下

subplots(nrows=1, ncols=1, sharex=False, sharey=False, ...)
Ex:
fig, (ax0, ax1) = plt.subplots(2,1)
ax0.plot(...)
ax0.set_title(...)
...

axes([left, bottom, width, height]) # 参数范围(0, 1)
\# plt.axes([....])会根据位置信息在绘制子图

#### 4.3 基于pandas的绘图

aDF.plot(fmt)

****

## Chapter 6 - 数据获取

### 1 基于内建模块的文件存取

#### 1.1 文件

**文件**是存储在某种介质上的信息集合，存取方式有顺序存取和随机存取，根据内容表示方式分为二进制文件和文本文件

**文本形式：**
-一个字节与一个字符对应
-便于对字符进行逐个处理，也便于输出字符
-占存储空间较多
-要花费转换时间

**二进制形式：**
-可节省外存空间和转换时间
-一个字节不对应一个字符，不能直接输出字符形式
-可读性差，常用于保存中间结果数据和运行程序

Python中可以处理二进制文件和文本文件，二进制文件的操作可以选择是否使用缓冲区，文本文件均使用缓冲区。文件的使用过程分为打开文件、文件读写、关闭文件。

#### 1.2 文件的打开/关闭

**文件的打开：**
file_obj = open(filename, mode='r', buffering=-1)
\# mode为打开模式，buffering大于等于1表示缓冲一行或指定缓冲区大小
\# 其他常用参数包括encodong等(encoding指定编码字符集)
\# open()返回一个文件对象，**文件对象可迭代**

open()函数 - mode

| Mode | function                                             |
| ---- | ---------------------------------------------------- |
| r    | 以读模式打开，文件必须存在                           |
| w    | 以写模式打开，若文件不存在则新建文件，否则清空原内容 |
| x    | 以写模式打开，若文件已存在则失败                     |
| a    | 以追加模式打开，若文件存在则向结尾追加内容，否则新建 |
| r+   | 以读写模式打开                                       |
| w+   | 以读写模式打开(清空原内容)                           |
| a+   | 以读和追加模式打开                                   |
| rb   | 以二进制读模式打开                                   |
| wb   | 以二进制写模式打开(参见w)                            |
| ab   | 以二进制追加模式打开(参见a)                          |
| rb+  | 以二进制读写模式打开(参见r+)                         |
| wb+  | 以二进制读写模式打开(参见w+)                         |
| ab+  | 以二进制读写模式打开(参见a+)                         |

**文件的关闭：**
fp.close() # fp为文件对象，切断文件对象与外存储器中文件之间的联系

```python
try:
    with open('filename', 'r') as fp:
        ... # process file
except IOError as err:
    print(err)
```

#### 1.3 文件的读写

**读文件方法：**
s = fp.read(size) 
\# 从文件当前位置读取size字节数据，若size为负数或空，则读到文件结束
\# 返回一个字符串(文本文件)或字节流(二进制文件)
s = fp.readline(size = -1)
\# 从文件当前位置读取本行内size字节数据，若size为默认值或大小超过当前位置到行尾字符长度，则读取到本行结束(包含换行符)
\# 返回读取到的字符串内容
lines = fp.readlines(hint = -1)
\# 从文件当前读写位置开始读取需要的字节数，至少为一行；若hint为默认值或负数，则读取从当前位置到文件末尾的所有行
\# 返回从文件中读出的行组成的列表

**写文件方法：**
fp.write(s)
\# 向文件中写入数据(字符串或字节流)
\# 返回写入的字符数或字节数
fp.writelines(lines)
\# 向文件中写入列表数据，多用于文本文件

文本读写中常用到文本对象可迭代的性质：
for line in fp

#### 1.4 文件的定位

fp.seek(offset, whence=0)
\# fp打开的文件必须允许随机访问
\# 在文件中移动文件指针，从whence(0表示文件头部，1表示当前位置，2表示文件尾部)偏移offset个字节
\# 返回当前的读写位置

### 2 基于pandas的文件存取

#### 2.1 DataFrame数据存取

**读csv文件：**
data = pd.read_csv('score.csv', encoding='gb2312')

**写csv文件：**
df.to_csv('score_copy.csv')

**读写excel文件：**
df = pd.read_excel('score.xlsx')
df.to_excel('score.xlsx', sheet_name = 'score')

pandas支持读写的文件格式：
read_ +: clipboard, csv, excel, fwf, gbq, html, json, msgpack,
pickle, asa, sql, sql_query, sql_table, stata, table

### 3 json格式文件存取

**json格式**(JavaScript Object Notation, JS对象标记)是一种轻量级的数据交换格式

Example:

```json
>>> x = {"name":"Niuyun",
     "address":{"city":"Beijing","street":"Chaoyang Road"}}
>>> x['address']['street']
'Chaoyang Road'
```

#### 3.1 json格式字符串转换

import json

json_str = json.dumps(dict) # 将字典转化为json格式字符串
dict = json.loads(json_str) # 将json格式字符串转化为字典

#### 3.2 json格式文件存取

```python
with open('data.json', 'w') as f:
    json.dump(data, f) # write
    
with open('data.json', 'r') as f:
    data = json.load(f) # read
```

dump, load相当于在dumps, loads的基础上添加了文件操作

### 4 网络数据爬取

用Python可以获取网络数据（抓取网页，解析网页内容）
**抓取：**
-Requests第三方库
-Scrapy框架
**解析：**
-Beautiful Soup库
-re模块

#### 4.1 Requests库

Requests库是更简单、方便和人性化的Python HTTP第三方库

import requests
r = requests.get('website') # 请求获取指定URL位置的资源，对应HTTP协议的GET方法，返回一个Response对象 

[to be finished]

#### 4.2 Robots协议

[to be finished]

#### 4.3 网络数据解析

[to be finished]

****

## Chapter 7 - 数据预处理

数据处理过程：
DATA --(selection)--> 
TARGET DATA --(preprocessing)--> 
PREPROCESSED DATA --(transformation)--> 
TRANSFORMED DATA --(data mining)-->
PATTERNS --(interpretation/evaluation)-->
KNOWLEDGE

### 1 数据清洗

#### 1.1 缺失值处理

**处理方式：**删除/填充

**填充方式：**固定值、均值/中位数/众数、上下数据、插值函数、最可能的值

scores_df = pd.read_excel('scores.xlsx', index_col='name')
df.isnull() # 判断缺失值
df.dropna() # 删除缺失行
df.fillna(method='ffill', inplace=True) # 填充缺失行，此处使用均值填充

#### 1.2 异常值处理

**观察异常值：**
简单统计，绘图，基于密度、最近邻和聚类等方法

**处理方法：**
删除，同缺失值处理，局部均值(分箱)，不处理

### 2 数据集成

实体识别
冗余属性/记录识别
数据值冲突的检测与处理

### 3 数据变换

解决了量纲不同、数值范围差异大的影响

#### 3.1 最小-最大规范化

$x' = \frac{x-min}{max-min}$

(df-df.min())/(df.max()-df.min())

from sklearn import preprocessing
min_max_scaler = preprocessing.minmax_scale(df) # [0,1]
max_abs_scaler = preprocessing,maxabs_scale(df) # [-1,1]

问题：
若将来的数字超过了min和max，会越界，需要重新定义
若某个数很大则规范化后值相近且均接近0

#### 3.2 z-score规范化

$x' = \frac{x-\bar{x}}{\sigma}$

(df-df.mean())/df.std()

特征：
使用最多
处理后数据均值为0，标准差为1

scaler = preprocessing.scale(df)

#### 3.3 小数定标规范化

$x'=\frac{x}{10^j}$

df/10**np.ceil(np.log10(df.abs().max()))

特征：
移动小数点位置，移动位数取决于属性绝对值的最大值
常见落在[-1, 1]之间

#### 3.4 连续属性离散化

方法：
分箱：等宽法、等频法
聚类

### 4 数据规约之数值规约

#### 4.1 数据规约

**目的：**对属性和数值进行规约，获得一个原数据集的小的规约表示，但仍接近原数据的完整性，在规约后数据集上挖掘可产生近乎相同的分析结果

**属性规约：**向前选择，向后删除，决策树，PCA
**数值规约：**有参方法(回归法，对数线性模型)，无参法(直方图，聚类，抽样)

#### 4.2 直方图

**表现：**
用分箱表示数据分布
每个桶（称为单桶）代表一个属性-频率对
数值范围差异大

#### 4.3 抽样

**抽样：**
随机抽样：不放回
iris_df.sample(n = 20)
iris_df.sample(frac = 0.3)
随机抽样：有放回
iris_df.sample(frac = 0.3, replace = True)
聚类抽样
分层抽样：数据集划分成互不相交的层，对每层进行简单随机抽样

****

## Chapter 8 - 数值计算与优化

数值计算指有效使用数字计算机求数学问题近似解的方法与过程，以及由相关理论构成的学科，主要包括：方程求解、矩阵特征值和特征向量求解、插值方法、最优化、数值积分...

### 1 方程求根与最小二乘法

#### 1.1 线性方程组求解

详见Chapter 5 - 2.4

#### 1.2 超定线性方程组求解

#### 1.3 最小二乘法

通过**最小化误差的平方和**寻找数据的最佳函数匹配，常用于**曲线拟合**。

[TO BE FINISHED]

### 2 极值与梯度下降

[TO BE FINISHED]

### 3 特征值分解与奇异值分解

[TO BE FINISHED]


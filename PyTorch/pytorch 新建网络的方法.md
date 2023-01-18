

# Pytorch定义网络的几种方法

## 直接申明式

首先，最简单的肯定是直接申明了

```python
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,10)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(10,2)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
```

这是最简单的定义一个网络的方法，但是当网络层数过多的时候，这么写未免太麻烦。

## nn.ModuleList()

应以上问题，Pytorch 还有第二种定义网络的方法 nn.ModuleList()

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = nn.ModuleList([nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2)])
    def forward(self,x):
        x = self.base(x)
        return x
```

nn.ModuleList()接收的参数为一个List，这样就可以很方便的定义一个网络，比如：

```python
base = [nn.Linear(10,10) for i in range(5)]
net = nn.ModuleList(base)
```

注意如果只使用python的list ， 不使用 nn.ModuleList(); 参数是不会注册到网络中的,如下：

```python
class net2(nn.Module):
    def __init__(self):
        super(net2, self).__init__()
        self.linears = [nn.Linear(10,10) for i in range(2)]
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x

net = net2()
print(net)
# net2()
print(list(net.parameters()))
# []
```



nn.ModuleList()是可以通过索引进行调用的,如下这个例子利用索引改变了调用的次序：

```python
class net3(nn.Module):
    def __init__(self):
        super(net3, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,20), nn.Linear(20,30), nn.Linear(5,10)])
    def forward(self, x):
        x = self.linears[2](x)
        x = self.linears[0](x)
        x = self.linears[1](x) 
        return x
```



==nn.ModuleList()是没有前向函数的，需要自己实现前向函数。而下面的nn.Sequential()是实现了前向的。==



## nn.Sequential()

另外一个方法就是nn.Sequential()

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.base = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
    def forward(self,x):
        x = self.base(x)
        return x
```

### OrderedDict

当然nn.Sequential()还有另外一种用法OrderedDict(有序字典)

```python
class MultiLayerNN5(nn.Module):
    def __init__(self):
        super(MultiLayerNN5, self).__init__()
        self.base = nn.Sequential(OrderedDict([
            ('0', BasicConv(1, 16, 5, 1, 2)),
            ('1', BasicConv(16, 32, 5, 1, 2)),
        ]))
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
```

### add_module

除此之外，nn.Sequential 还能 add_module

```python
class MultiLayerNN4(nn.Module):
    def __init__(self):
        super(MultiLayerNN4, self).__init__()
        self.base = nn.Sequential()
        self.base.add_module('0', BasicConv(1, 16, 5, 1, 2))
        self.base.add_module('1', BasicConv(16, 32, 5, 1, 2))
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
```

## nn.Sequential()以及nn.ModuleList()的区别

来看一下nn.Sequential()以及nn.ModuleList()的主要区别，我个人感觉就是nn.Sequential()里面自带了forward函数，可以直接操作输入，而nn.ModuleList()需要定义一个forward函数。

```python
tt = [nn.Linear(10,10), nn.Linear(10,2)]
n_1 = nn.Sequential(*tt)
n_2 = nn.ModuleList(tt)
x = torch.rand([1,10,10])
x = Variable(x)
n_1(x)
n_2(x)#会出现NotImplementedError
```

在定义比较深的网络的时候，结合nn.ModuleList()以及nn.Sequential()在代码量上会看上去十分简洁。

## 继承 nn.Sequential

最近在看denseNet的时候，又学到了一种定义网络的办法，就是直接继承nn.Sequential

```python
class DenseLayer(nn.Sequential):
    def __init__(self):
        super(DenseLayer, self).__init__()
        self.add_module("conv1", nn.Conv2d(1, 1, 1, 1, 0))
        self.add_module("conv2", nn.Conv2d(1, 1, 1, 1, 0))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)
#这个写法和下面的是一样的
class DenLayer1(nn.Module):
    def __init__(self):
        super(DenLayer1, self).__init__()
        convs = [nn.Conv2d(1, 1, 1, 1, 0), nn.Conv2d(1, 1, 1, 1, 0)]
        self.conv = nn.Sequential(*convs)
    def forward(self, x):
        return torch.cat([x, self.conv(x)], 1)
net = DenLayer1()
x = torch.Tensor([[[[1, 2], [3, 4]]]])
print(x)
x = Variable(x)
print(net(x))
```
















# PyTorch 数据集随机值的完美实践




来源 | https://zhuanlan.zhihu.com/p/377155682



本文所分析的问题与解决方案将在最近发布的pytorch版本中解决；因此解决所有烦恼的根源是方法，更新pytorch

一个快捷的解决方案：

```
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

## 01 关于pytorch数据集随机种子的基本认识

在pytorch中random、torch.random等随机值产生方法一般没有问题，只有少数工人运行也可以保障其不同的最终值.

np.random.seed 会出现问题的原因是，当多处理采用 fork 方式产生子进程时，numpy 不会对不同的子进程产生不同的随机值.

换言之，当没有多处理使用时，numpy 不会出现随机种子的不同的问题；实验代码的可复现性要求**一个是工人种子** ,即工人内包括numpy，random，torch.random所有的随机表现；**另一个是Base** ,即程序运行后的初始随机值，其可以通过以下两种方式产生

1. torch.manual_seed(base_seed)
2. 由特定的seed generator设置

```
generator = torch.Generator()
g.manual_seed(base_seed)
DataLoader(dataset, ..., generator=generator)
```

使用spawn模式可以斩断以上所有烦恼.

## 02 直接在网上搜这个问题会得到什么答案

参考很多的解决方案时，往往会提出以下功能：

```
defworker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
```

让我们看看它的输出结果：
（第0，3列是索引，第1，4列是np.random的结果，第2，5列是random.randint的结果）

```
epoch 0
tensor([[    0,  5125, 13588,     0, 15905, 23182],
        [    1,  7204, 19825,     0, 13653, 25225]])
tensor([[    2,  1709, 11504,     0, 12842, 23238],
        [    3,  5715, 14058,     0, 15236, 28033]])
tensor([[    4,  1062, 11239,     0, 10142, 29869],
        [    5,  6574, 15672,     0, 19623, 25600]])
============================================================
epoch 1
tensor([[    0,  5125, 18134,     0, 15905, 28990],
        [    1,  7204, 13206,     0, 13653, 25106]])
tensor([[    2,  1709, 15512,     0, 12842, 29703],
        [    3,  5715, 14201,     0, 15236, 27696]])
tensor([[    4,  1062, 13994,     0, 10142, 23411],
        [    5,  6574, 18532,     0, 19623, 21744]])
============================================================
```

假设上述方案对一个时代内可以防止不同的工人出现随机值相同的情况，但**不同的时代之间，其最终的随机种子仍然是不变的。**

## 03 那应该如何解决

### 来自pytorch官方的解决方案：

https://github.com/pytorch/pytorch/pull/56488#issuecomment-825128350

```
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
 
ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

### 来自numpy.random原作者的解决方案：

https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

```
def worker_init_fn(id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
 
ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

### 一个更简单但不保证正确性的解决方案：

```
def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)
```

### 04 附上可运行的完整文件

```
import numpy as np
import random
import torch

# np.random.seed(0)

class Transform(object):
    def __init__(self):
        pass

    def __call__(self, item = None):
        return [np.random.randint(10000, 20000), random.randint(20000,30000)]

class RandomDataset(object):
    def __init__(self):
        pass

    def __getitem__(self, ind):
        item = [ind, np.random.randint(1, 10000), random.randint(10000, 20000), 0]
        tsfm =Transform()(item)
        return np.array(item + tsfm)
    def __len__(self):
        return20

from torch.utils.data import DataLoader

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

ds = RandomDataset()
ds = DataLoader(ds, 10, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn)

for epoch in range(2):
    print("epoch {}".format(epoch))
    np.random.seed()
    for batch in ds:
        print(batch)
```




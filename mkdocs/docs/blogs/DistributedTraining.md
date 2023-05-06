---
title: 用PyTorch进行分布式的模型训练
---
如果想要做更多的事，分布式训练是必不可少的，尤其在LLM成为热点的今天。这篇Blog基于Torch的官方分布式教程，简单的介绍Torch中的分布式训练。

## Dataparallel 和 DistributedDataParallel

### nn.DataParallel()
采用`nn.DataParallel()`可以使用最少的代码实现单机上的多卡分布式训练，假设我已经实例化了`model`，并准备好了那么只需要在将模型放在device前采用`DataParallel()`包装一下即可：

=== "通常"
    ```py title="Seq2Seq.py" linenums="1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ```
=== "使用nn.DataParallel()包装"
    ```py title="Seq2Seq.py" linenums="1"
    device_lis = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    model = nn.DataParallel(model, device_ids=device_lis)
    ```
!!! note 
    这种方式是单进程多线程的，由于Python的GIL，这实现的并非真正意义上的多线程。

### nn.parallel.DistributedDataParallel()  
官方的Tutorials极力推荐使用它替代上述的`nn.DataParallel()`，原因有以下三点:

- `DataParallel()`只适用于在单机多卡的情况，而`DistributedDataParallel()`还适用于多机多卡；
- `DataParallel()`用单进程+多线程实现分布式训练，而`DistributedDataParallel()`用多进程实现分布式训练；
- `DistributedDataParallel()`支持将模型的实例的不同层放在不同的GPU上，而`DataParallel()`只能将模型的实例都放在相同的GPU上。

## 实用代码
```py title="引入库"
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
```
PyTorch在官方[Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)中提供了三个例子，最基础的使用方式需要包括三个基本步骤：

1. 添加环境变量并初始化进程组:
```py
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
```
这里的`rank`和`world_size`需要加以解释，`rank`表示全局的进程编号，`world_size`表示单台主机上的进程数，例如我一共有两台主机，每台主机上有四张GPU，并且一张GPU都足够我放下`model`和训练的数据，那么我希望为每张显卡都创建一个进程，并为它们从0开始进行编号，那么就一共有`0, 1, 2, ..., 6, 7`八个进程，这里的编号就是`rank`，而`world_size`则是每台主机上创建的进程数，它等于每台主机上的GPU数量。  
2. 初始化模型`model`并使用`DistributedDataParallel()`包装，假设一个输入`data`和对应的标签`label`:
```python
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    # 调用刚才定义好的setup函数并初始化模型
    setup(rank, world_size)

    # 用DDP包装模型，并指定当前模型所在的进程号(设备号)
    model = ToyModel()
    ddp_model = DDP(model, device_ids=[rank])

    # 定义loss_function和optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # 将data和labels放在当前进程的GPU上
    data, label = data.to(rank), label.to(rank)
    outputs = ddp_model(data)

    # 反向传播和更新参数
    loss_fn(outputs, label).backward()
    optimizer.step()

    # 释放进程组
    dist.distroy_process_group()
```
这里demo_basic函数中的`model`也就是继承`nn.Module`后的类的实例。在前面我们提到了`DistributedDataParallel`也适用于一个模型放置在多张卡上的情况，这需要在定义模型的类时，在`__init__`函数中添加`dev`参数，Torch官方给出了两种示例：    
```python title="单张卡上的模型定义示例"
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
```
```py title="多张卡上的模型定义示例"
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```  
对比可以发现，多张卡上的模型将`self.net1`和`self.net2`分别定义在了`dev0`和`dev1`上，事实上就是两张不同的卡的device_id，并且在`forward`函数中，输入`x`最开始在`dev0`上进行处理，最终的输出是在`dev1`上的。因而针对需要放在多张卡上的模型，需要为它指定dev0和dev1具体的`local rank`:
```py linenums='1'
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    # 调用刚才定义好的setup函数并初始化模型
    setup(rank, world_size)

    # 为多张卡上的模型指定dev0和dev1，local rank 需要对world_size取模
    dev0 = (rank * 2) % world_size
    dev1 = (rank * 2 + 1) % world_size

    # 用DDP包装模型，注意与单卡模型不同的是不要指定device_ids
    model = ToyMpModel(dev0, dev1)
    ddp_model = DDP(model)

    # 定义loss_function和optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # 将data和labels放在对应的GPU上
    data, labels = data.to(dev0), labels.to(dev1)
    outputs = ddp_model(data)

    # 反向传播和更新参数
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # 释放进程组
    dist.distroy_process_group()
```
可以看到上述代码中实际上是将rank连续的为偶数和奇数的两块GPU分别作为`dev0`和`dev1`，并且输出的`outputs`也是在`dev1`上与`label`一起参与loss的计算。  
3. 采用spawn方式启动进程:
```py
if __name__ == "__main__":
    wold_size = torch.cuda.device_count()
    mp.spawn(demo_fn=demo_basic,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

## DDP模型的保存和加载
DDP的并行是通过不同进程之间维护一个共同的optimizer实现的，如果每张显卡负责一个进程，由于每张显卡上的模型参数都是一样的，因此只需要保存一份模型参数即可，即假定我已经指定模型的checkpoint路径为`model_path`，那么只需要在主进程中保存模型参数即可，即在`demo_basic`函数中加入如下代码:
```py

```




## 参考资料
[GETTING STARTED WITH DISTRIBUTED DATA PARALLEL](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)


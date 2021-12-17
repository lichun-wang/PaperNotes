# Benchmarking detection tranfer learning with vision transformers

>

---

Author: Yanghao Li, KaimingHe, FAIR

code: will release in Detectron2

---

## Abstract

- 目标检测是一个比较重要的下游任务，来验证pre-trained model的有效性。特别是当一个新的网络被提出的时候。
- 存在一些困难，比如：模型不兼容、训练比较慢、显存太高、缺乏训练参数等问题导致vit在detection的应用止步不前。
- 本文克服上面的困难，将vit应用到mask-rcnn中
- 实验证实，最新的基于mask的预训练方法，取得了非常不错的效果。在COCO上起码有4%的提升

## Introduction

## Approach

![image-20211202103755344](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211202103755344.png)

### Vit Backbone

- 如何在ViT中引入FPN结构

  >FPN需要multi scale的，但是vit是single scale的，如何解决？
  >
  >1. 采用XCiT中提到的方法，如何做的呢？如上图所示，对4个中间的block输出，采用上采样或者下采样来构建出4个不同的尺度，分别是4倍，2倍，1/2倍
  >2. upsamples采用的是2x2的 transposed convolution ，第一个还会加上group normalization和GeLU，其他不加，最后的downsample采用的是2x2的max pooling
  >3. 上述操作不改变网络的维度。
  >4. 作者还提出最近的比如swin，采用修改vit网络的方式，使用窗口的聚合操作形成天然的multi scale,这个方法也是不错的方法，但是其由于修改了原始的vit，使其更复杂，不利于unsupervise learning的应用。

- 如何在使用ViT的时候降低显存

  > 1. 在目标检测中，输入的尺寸往往比较大，可能是1024x1024,这样会导致计算量飙升，在mask-rcnn里面用了base的vit都需要20-30G的显存。
  > 2. 为了降低显存，这里使用了windows来划分self-attention的作用范围，但是使用windows会来带一个问题就是不同窗口会缺乏交互，所以，这里作者采用的方法是，将所有的block划分成4份，每份的最后加上一个全局的self-attention,
  > 3. 也就是这4个全局的self-attention层，对应的就是接上面的FPN的结构，如上图所示。

### Upgraded Modules

- 在FPN中，conv后面接了BN层
- RPN中使用2个conv，取代原来的1个
- 分类头和回归头使用4个conv接fc，而不是2个fc

### Training Formula

- 当模型训练的特别多epoch的时候（400），使用large-scale jitter(LSJ)可以防止过拟合。==就是随机改变输入尺寸==

- drop path regularization, 就是随机深度

- 总体配置：

  > 1. LSJ : [0.1, 2.0], 1024x1024
  > 2. AdamW , 0.9, 0.999
  > 3. cosine learning rate decay
  > 4. linear warmup 0.25epoch
  > 5. drop path regularization(随机深度)
  > 6. finetune 100 epochs, from scratch: 400 epoch
  > 7. 32-64 gpus V100-32G, minibatch=64 

### Hyperparameter Tuning Protocol(调哪些超参数呢)

- 只调三个参数，其他都是不变的：lr + weight decay + drop path rate

- 咋调呢？

  > 1. ViT-B 针对lr以及weight decay, 采用3x3的中心搜索，初始lr和wd为1.6e-4 ,0.1,然后在half和double中进行搜索，如果没找到最优就扩大范围
  > 2. dp在【0， 0.1， 0.2.，0.3】中搜索，最终0.1效果比较好。
  > 3. ViT-L 默认采用ViT-B的lr 和 wd , 因为这个搜索太费时间了，然后dp=0.3效果比较好
  > 4. 当然作者也说了这种方式存在一定的限制。

### 更多的实现细节

- 在训练过程中，并没有采用force resize的方式将图像变成1024x1024, 而是采用的padding，但是在inference的时候，其实没有必要都padding到1024x1024，只要是patch的整数倍就可以了，这样还可以省时间，但是作者测下来，掉点了，所以inference作者也采用了padding到1024x1024了。

## Initialization Methods

- 5种初始化方式

  > 1. 随机初始化
  > 2. supervised: ImageNet 1K, DeiT的参数
  > 3. moco v3参数
  > 4. beit 参数（自己重新训的，official拿不到）
  > 5. MAE参数

- 但也存在一个问题是，这5个初始化本身也是很难做到公平的，比如可能不同的初始化方式需要不同的epoch来收敛， beit使用了DALL.E 的dVAE，其他的没有使用等。

## Exp

一、如下是各种不同的初始化方法的结果：可以看到MAE效果是最好的， moco v3比random的效果要差一些，跟supervised的结果比较接近，这也是挺奇妙的；换到ViT-L上面，都有一定程度提高，MAE和BEiT效果比较接近。

![image-20211202163924420](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211202163924420.png)


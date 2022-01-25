# Patches are all you need

> 这篇论文的创新点在于，将patch的思想应用的卷积中去，主要做法就是在图像上，先用一个patch_embedding来进行patch的提取，然后再进行各种卷积的操作（depth wise, point wise）,并且在整个网络中，作者没有采用downsample了，最终取得了还不错的效果。
>

---

github: https://github.com/tmp-iclr/convmixer.

---



## Abstract

- vit为了降低计算复杂度以及参数量，采用了patch的方式，取得了不错的效果，那么这个不错的效果究竟是transformer结构带来的，还是采用了patch这种方式带来的呢？
- 本文主要针对后者，提取patches进行探索，提出了convMixer，尽管简单，但是效果也outperform了些许模型
- 

## Introduction

![image-20211112104115656](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112104115656.png)



## ConvMixer

- patch embedding的过程：就是用卷积大小和patch大小相同的卷积 进行维度变换。

  

![image-20211112104800666](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112104800666.png)

- ConvMixer block: 就是用depthwise+pointwise + identity, 经验来看depthwise的kernel size 要大一些。

  ![image-20211112105201384](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112105201384.png)


- 结构和代码如下：

![image-20211112105641399](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112105641399.png)

![image-20211112105809264](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112105809264.png)

## Motivation

- 本文的思路来源于mix-mlp，只不过本文是卷积运算，卷积的感受野受限，所以不能像self-attention或者mlp那样具有全局的感受野，所以在depth-wise conv中，尽量使用大的卷积核，提高感受野。

## Experiments

- 已经加入了timm
- using randaugment, mixup,cutmix, random erasing, gradient norm clipping
- use AdamW
- 实验表明，减少kernel size ,会掉点大约1%，如果采用更小的patches，会取得更好的效果，作者认为更大的batch需要更深的网络。
- ![image-20211112140503099](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211112140503099.png)

## 附录


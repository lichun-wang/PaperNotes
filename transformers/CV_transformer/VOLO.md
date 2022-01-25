# VOLO:Vision Outlooker for Visual Recognition

>这篇文章的最大创新点在于，设计了一个不同于self-attention的attention-outlook attention，不仅仅看到当前点，可以计算临近的K\*K的滑窗内的attention，并且在计算outlook attention的时候，采用了更小的patch,作者的意思是可以获得更精细的信息，通过加入outlook attention ,最终只在imagenet上取得了超过conv的效果

---

Author:  Li Yuan , Qibin Hou  , 颜水成团队

Github: https://github.com/sail-sg/volo

---

## Abstract

- 如果没有额外的数据，一般transformer很难干过cnn
- 为啥vit不行，原因是vit对精细的特征（fine-level feature）到token的编码的低效导致的。
- 提出了一个新的outlook attention 方法
- no extra data,  imagenet, 87.1% acc
- downstream tasks

![image-20211118104506819](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211118104506819.png)

## Introduction

- Outlooker的创新点方式在于 对token的聚合进行attention操作，使得模型可以有效的encode 细级别信息。
- 通过高效的线性投影，来直接聚合token附近token，这样摆脱了较为昂贵的dot-product attention的计算。

## Method

- volo包含两个stage,第一个stage使用outlooker来聚合token的表示信息，第二个stage使用transformer blocks来聚合全局信息

- Outlooker attention

  ![image-20211118111433804](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211118111433804.png)

### Outlook Attention

- outlook attention主要的insight如下两点：

  > 1. 每一个空间位置特征是足够聚合其局部相邻特征的（the feature at each spatial location is representative enough to generate attention weights for locally aggregating its neighboring features;）
  > 2. 局部空间的聚合可以更有效的编码fine-level信息。

- 具体是怎么做的呢？简单来说，就是通过计算每个token相邻的$k*k$个token的相似度作为attention，而不是使用self-attention这种方式了。减少计算量。具体是怎么做的呢？如下说明。

  > 1. 对于输入尺度为][H, W, C]的图像X， 首先通过linear分别映射到 C和$K^4$的空间，得到的V和Att分别为[H, W,C]和[H,W, $K^4$],为什么是$K^4$呢？因为其相邻的是K\*K个，然后每个是算全部的相关性就又是K\*K个，总共就是$k^4$了
  > 2. 对于V,通过torch.unfold操作，提取局部窗口，得到[H,W,K\*K,C]的矩阵，然后att通过reshape得到[H,W,$K^2$,$K^2$]的矩阵，对att求softmax，就得到[H,W,$k^2$,1]的矩阵，就是atten了，就可以跟上面的V相乘了
  > 3. 然后在torch.fold回去
  > 4. 这也就对应了attention的名字，**outlooker**,
  >
  > ![image-20211118174024858](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211118174024858.png)
  >
  > ![image-20211118174258054](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211118174258054.png)
  > 

### Multi-Head Outlook Attention

- 这个多个head是怎么干的呢？有了上面的基础，其实也比较简单：

- >1. 首先对于V，其维度是[H, W, C, K\*K],这里，我们将C划分成N份就可以了，而且每一份的长度其实是可以不一样的。
  >2. 对于Att，其维度是 [H,W,$K^2$,$K^2$]，变成[H,W,N\*$K^2$,$K^2$] 就可以了。

  ### Discussion

- ==相比于卷积，这种计算token相似度的方法，可能更有助于特征学习时候提高参数效率。==这是真的吗？
- 我们采用的slide window 的方式 融合了更多的位置信息
- 我们生产attention的方式很快，而且有效

  ## Network Architecture Variants

- based on LV-ViT
- 网络分为两个阶段，第一个阶段使用8x8的patch，利用outlook attention 提取fine level information，然后第二个阶段对patch进行合并，利用transformer来进行信息整合。
- Outlooker and Transformer的比例大约是1：3
- 在最后的stage，加了两个class attention layer来升级class embedding
- Outlooker的hidden dimension 设置为了transformers的一半。

## Experiments

- **Only ImageNet dataset**
- 《Token labeling for training better vision transformers》 这篇论文可以看看是啥，这篇里面用了token labeling tookit

### setup

> adamW 
>
> lr = Lr \*  b_s/ 1024  , weight_decay:0.05
>
> Stochastic  Depth
>
> train 300 epoch
>
> data augmentation: cutout , randaug, token labeling objective with mixtoken,   not use mixup and cutmix
>
> VOLO-D1, VOLO-D2 , 4块GPU就可以训起来了
>
> 在大分辨率finetune， lr = 5e-6,  weight decay=1e-8  , 30 epoch
>
> ![image-20211118193738422](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211118193738422.png)



### Ablation Analysis

- Model Scaling: 1个是改网络深度， 2. 提高图像分辨率， ==两种方法都可以比较稳定的提高模型的表现==
- outlooker的数量： 在4个的时候比较好，再大就没有明显提升了。

### Semantic Segmentation

- UperNet


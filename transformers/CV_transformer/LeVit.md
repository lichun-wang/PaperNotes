# LeVit：a Vision Transformer in VonvNet's Clothing for Faster Inference

>本篇论文是facebook ai research提出的一个速度与精度平衡的vision transformer网络，论文的主要创新点在于融合了卷积以及transformer，并且将transformer中linear全部使用1x1的卷积进行了等价替代（不清楚这个等价替代能带来多大的精度优化），并且设计了类resnet结构的网络，同时跟vit类似，也是采用的双头设计，即分类头以及蒸馏头，总体来看，这篇论文更像是将self-attention的conv变种应用到卷积网络中，导致网络不再像transformer了，最终论文表示，gpu速度达到了efficientnet的2倍以上，同时精度还超过了efficientnet，值得去试一试。

---

**Author**: Ben Graham

**Facebookresearch**

**Github**: https://github.com/facebookresearch/LeViT.

---

## Abstract

- fast inference classification model
- attention bias ， a new way to 替代 position information.
- 80% imagenet top-1 acc, 在cpu上LeVit是efficinetNet的5倍速度，gpu差不多是2倍。

## Introduction & Related Work

- 为啥较LeVit，因为用了形状想LeNet的Vit，所以较LeVit

- 卷积往往需要更多的访问内存，所以会带来速度的减慢，self-attention天然就应该会更快一些。

- 本文创新点，压缩vit模型，平衡分辨率和宽度：

  > - 下采样策略
  > - 一个高效的patch描述方法
  > - 替换positional embedding
  > - 重新设计了attention-mlp，改善网络容量

## Motivation

### Convolutions in the vit

- vit是通过 nn.Linear来进行patch embedding的，到了deit后就变成了通过conv（16x16的核）来进行patch embedding了
- 但是在qkv的计算上，vit或者deit使用的依然是传统的linear， levit这里改成了也可使用conv来进行投影。
- 卷积可以带来空间的平滑性（smoothness）,因为有overlap，但是vit的卷积没有overlap所以平滑性靠的是data augmentation

### Preliminary experiment:grafting(嫁接)

- 这里研究一下在相似计算量的情况下，将conv与conv嫁接到一起
- 嫁接ResNet-50 和 DeiT-Small, 去ResNet的头部和Deit的后部， conv先到14x14，然后接Deit，中间加了pooling层，position embedding加载了Deit上面。
- 结果如下，发现嫁接最好的是2个resnet stage,6层transformer;相比存的resnet和纯的deit效果都要好一些。
- ![image-20211105142246346](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211105142246346.png)
- 并且发现，训练的时候，前期更像是conv 后期更像是transformer

## Model

### Design principles of LeVit

- 大火箭结构

![image-20211105145537787](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211105145537787.png)

###   Components

- **Patch Embedding** 前面接4个conv（3x3）,  channel分别为：3,32,64,128,256，输出feature map大小为：（256， 14， 14）
- **No classification token**  类似conv，使用avg pooling然后接分类层， 并且采用了==分类头和蒸馏头==，两个头。inference的时候，两个头进行融合。
- **Normalization layers and activations**:  將vit中的fc layer都用了卷积进行替代。
- Multi-resolution pyramid： 类似resnet50那种结构
- Downsampling: 类似卷积，特征图减小，channel增加
- **Attention bias instead of a positioonal embedding**: 这个挺有意思的。需要详细体会下。意思就是，我在attention的内部，集成位置信息，而不是在最开始的时候加入位置信息，参考论文《Do we really need explicit position encodings for vision transformers》
- Smaller keys: 限制了K和Q的大小之后，可以减少$QK^T$的时间
- Attention activation: Hardwish

![image-20211108104934890](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108104934890.png)



![image-20211108114104779](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108114104779.png)



## Experiments

- 32GPU，训练1000 epochs

- 同时使用两个分类头，一个是label， 一个是 distillation

- 对比效果，比efficientnet要快很多：结果一目了然。

  ![image-20211108114130584](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108114130584.png)

* 消融实验，到底增加的哪个模块的作用最大呢？

  ![image-20211108142512086](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108142512086.png)


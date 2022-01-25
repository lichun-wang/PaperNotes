# SimSiam

> 这篇文章，作者表示，simsiam这个结构其实对目前这些无监督的对比学习方法产生了比较大的作用，作者设计了一个simsiam的结构，既没有momentum，也没有大的batchsize，采用了stop-gradient的方法，也取得了相当不错的效果。文中的假设我每太看懂，待后续再探索吧。



analogous 类似

repluse 

symmetrized

Symmetrization



---

## Abstract

- siamese network 在自监督中应用比较普遍，这些model一遍都是最大化同一张图像的两个augment，同时避免训练collapse
- 我们的实验证明，即使不使用 [ 负样本对，大batch size ， momentum encoders ] 依然可以学到好的特征表达。
- 我们的实验证明，stop-gradient 对于避免 collapsing  很重要。

## Introduction

- siamese 网络 最怕的就是所有的结果都输出一个常数
- 我们的方法==直接最大化 两张图像不同view的相似性==。既不用负样本，也不用momentum encoder，而且是正常的batch size (不大)
- We hypothesize that there are implicitly two sets of variables, and SimSiam behaves like alternating between optimizing each set
- Siamese结构可能是目前这些方法成功的关键
- stop gradient 操作很重要

![image-20220120160644611](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120160644611.png)



![image-20220120164116781](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120164116781.png)



## Related

- momentum不是避免collapsing的关键，stopd-gradient 才是。

## Method

- two views of same image by different augmentation
- encoder shares weights
- loss: 这个跟byol中的loss是等价的

![image-20220120162851951](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120162851951.png)

- 本文采用了stopgrad,结果loss就变成了如下：==就是说，只有经过了predictor才会产生梯度，没有经过predictor是不会产生梯度的==

  

![image-20220120163522609](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120163522609.png)

- Baseline setting:

  > 1. SGD , not use LARS(large-batch)
  > 2. lr = 0.05, cosine schedule , weight decay 0.0001, momentum 0.9
  > 3. bs = 512; BN is used
  > 4. Projection MLP , 3 layers, has BN(包括输出的fc也有bn)， 输出fc没有relu, hidden fc : 2048d, 
  > 5. predictor, 2 layers, output=2048, hidden=512,   output no BN.

## Empirical Study

- 下表对比了，有无stop gradient的效果， 当没有stop-grad的时候，loss快速的到达了-1，输出趋于一致，knn基本失效，说明出现了collapsing，侧面说明了，predictor, BN 等都不是决定collapsing的关键。

![image-20220120170607919](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120170607919.png)

- predictor的研究：

  > 1. 没有predictor的时候是不能工作的，比如将predictor设置成 恒等。
  > 2. predictor 随机固定住也是不行的，训练不收敛，loss一直很大，不是collasping
  > 3. ==predictor lr not decay, 往往会带来更好的结果==，A possible explanation is that h should adapt
  >    to the latest representations, so it is not necessary to force it converge (by reducing lr) before the representations are sufficiently trained。
  > 4. 

  ![image-20220120172240215](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120172240215.png)

- Batch Size 的研究：

  ![image-20220120173855417](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120173855417.png)

- BN

  > 没有bn 效果不行 ，但是没有 collapsing
  >
  > c，效果比较好
  >
  > d,在最后的输出也加上bn，效果不行了，我觉得是因为最终的输出参产生了干扰，一般最终的输出都不会加BN

![image-20220120173930420](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120173930420.png)

- Similarity Function,如果将我们用的相似度判别loss，直接换成plogp这种，crossentropy 的分类loss，结果会掉点

  ![image-20220120174755816](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120174755816.png)

  

- Symmetrization

  采用对称和非对称的结构，发现都能训出来，==不过这里我有个疑问，当采用非对称结构的时候，没法进行参数共享了，那另一个encoder由于没有梯度，莫非是不更新的吗？==

  

![image-20220120175604205](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220120175604205.png)



## Hypothesis

这里过于理论，没太看懂。

- ==如果vector经过了l2-normalized,那么cosine similarity 和 mean squared error是等价的==




# Deep networks with stochastic depth

>

---

Github: https://github.com/yueatsprograms/Stochastic_Depth

---

## Abstract

- 深度卷积往往会带来效果的提升，但是深度网络也会带来一定的问题：比如：梯度消失，训练时间长等。
- 本文提出了一个stochastic depth的方法，一个看起来矛盾的设置：训练时候采用小网络，然后inference时候采用大网络。
- 主要方法就是，在训练的过程中，自动的扔掉一部分的layer，用identity来代替
- 该方法，可以有效的减少训练时间，同时实验证明，可以明显的提升模型ACC。

## Introduction

- similar to Dropout, training with stochastic depth acts as a regularizer

## Deep Networks with Stochastic Depth

- 对于每一个mini-batch,随机的选择一些层丢弃，用identity skip 来进行替代
- 下面这个公式就是本文的关键，但b=0时，就是短路，b=1时，就是正常的residual block
- ![image-20211123171959222](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211123171959222.png)
- 如何设置屏蔽的比例呢？本文采用了一种线性方式：为啥这么干呢？作者考虑浅层特征可能会被深层layer用到，所以对浅层保留了更多的layer
- ![image-20211123173010085](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211123173010085.png)
- 按照上面的参数，大约保留了3/4的layers。
- 这个方式跟模型ensemble有点像。
- 测试的时候，就按照原始的模型进行测试就可以了。


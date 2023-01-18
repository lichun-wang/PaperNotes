# Mask CTC

> Author : Yosuke Higuchi
>
> Year: 2020
>
> github:espnet

## Abstract

* non-autoregressive end to end asr.  自回归的网络下一个输出依赖于前一个输出，输出长度为L的结果需要L iter ,非自回归网络解码每个字是独立的，不相互依赖的，比如MLM的bert，非自回归会更快速
* joint training  of mask prediction and ctc
* inference: greedy ctc output and then low-confidence tokens are masked to decoder
* on wsj, mask-ctc 与标准ctc相比较，表现更优： wer： 17.9%-> 12.1%

## Introduction

* mask方式对长文本不好，因为前面很可能预测错误，这会对后面的也产生一定的影响
* No requirement for output length prediction
* Accurate and fast decoding

## Mask CTC framework

### Attention-based encoder-decoder

* 后一个token的产生依赖于前面的token的概率，如下图公式

<img src="..\images\image-20210603101442695.png" alt="image-20210603101442695" style="zoom:80%;" />

### CTC

* $\beta_{-1}(Y)$ 是所有可能的对其方式， CTC在所有token + blank

<img src="..\images\image-20210603101718727.png" alt="image-20210603101718727" style="zoom:80%;" />

![image-20210603105443977](..\images\image-20210603105443977.png)

### Joint CTC-CMLM non-autoregressive ASR

* 训练的时候，gt被randomly replaced by <mask>，<mask>的数量是1-L的随机采样的均匀分布，参考论文《Mask-predict: Parallel decoding of conditional masked language models》

* 单单使用cmlm进行语音识别，==效果不好，会存在跳过和重复的token输出==。而这恰巧是CTC可以解决的问题。

* 来源于灵感《Joint CTC-attention based end-to-end speech recognition using multi-task learning》，是不是可以联合CTC来进行训练，解决上面的问题？==CTC可以提供绝对的位置信息。==（原文是这么说的：we found that jointly training with CTC similar to [8] provides the model with absolute positional information (conditional independence) explicitly and improves the model performance reasonably well）

* 联合训练的loss:

  ![image-20210603111623971](..\images\image-20210603111623971.png)

### Mask CTC decoding

* 非自回归的方法进行预测的时候，必须都知道output的长度才能进行并行化的预测。如何得到长度呢？可以训练一个模型来预测长度，或者加一个token <length>,但是在语音识别里面，说话者的语速，频率都相差比较大，所以不太好预测。

* 利用beam search CTC以及language model 可以得到比较好的表现，只是会导致解码速度慢一点，影响非自回归的效率。

* 最终我们怎么做的呢？使用“greedy” CTC得到初始结果，然后利用cmlm decoder进行纠正，cmlm输入的是根据CTC的结果，对低分进行mask

  <img src="..\images\image-20210603114600861.png" alt="image-20210603114600861" style="zoom:50%;" />

  

* decoding methods: easy-first,  每次mask的数量 C = L/K， 《An efficient algorithm for easy-first non-directional dependency parsing》，当然后面实验也指出，当迭代与mask的个数相同时，即每个iter只是mask一个位置，效果最好。

  



## Experiments

* SpecAugment
* 80mel-scale filterbank
* 4  attention heads, 256 hiddeen units , 2048 feed-forward inner dimension
* encoder: 12 layers, decoder 6 layers
* 200-500 epochs
* mask CTC is 116 倍速度高于自回归方法。
* ==缺点：mask-ctc由于在进行MLM的时候，长度固定的，所以不能进行增加和删除操作，只能修改操作。==
* 从下表可以看出，mask-ctc其实效果并没有提升，但是速度却有很大的进步。

<img src="..\images\image-20210603140534978.png" alt="image-20210603140534978" style="zoom:50%;" />



## 值得关注的引用论文

《Joint CTC-attention based  end-to-end speech recognition using multi-task learning》

## 


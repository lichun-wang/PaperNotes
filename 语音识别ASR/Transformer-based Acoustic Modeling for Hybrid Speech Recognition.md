# Transformer-based Acoustic Modeling for Hybrid Speech Recognition

> Yongqiang Wang
>
> 2019

## Abstract

* transformer-based acoustic models
* 主要讨论了： various positional embedding , an iterated loss
* 研究了使用限制的上下文信息的transformer，使得流式应用成为可能。
* sota

## Introduction

* RNNs有很多限制：1.  RNN的输入不能很长  2.不能并行化
* 为了解决这些限制提出了一些网络： TDNN《A time delay neural network
  architecture for efficient modeling of long temporal contexts》，FSMN《Feedforward sequential
  memory neural networks without recurrent feedback》
* Self-Attention

## Hybrid Architecure

* the hybrid approach is admittedly less appealing as it is not end-to-end trained， 但是仍然是表现最佳的系统。

## Acoustic Modeling Using Transformer

### Self Attention and Multi-Head Attention

### Architecture of Transformer

![image-20210607172154844](..\images\image-20210607172154844.png)

### Positional Embedding

* 在原始的transformer中，利用sincos进行绝对位置的编码，来保证输出的顺序问题。但是在语音信号中，可能相对位置可能会更有用。
* Sinusoid positional embedding
* Frame stacking: 将n个上下文堆叠在一起 $x_t = (x_t,x_{t+1,...})$
* Convolutional embedding,《Transformers with convolutional context for asr》

### Training Deep Transformers

* 深网络不好训，use iterated loss 《Deja-vu: Double feature pre-sentation and iterated loss in deep transformer networks》

### Relation to Other Works

* original transformer use  self-attention and cross-attention, this paper only use self-attention ,no cross-attention
* 

## Experiments

### Experiment Setups

* 1-state HMM  topology with fixed self-loop and forward transition probability (both 0.5)
* 80d log mel-filter bank features , 10ms
* 降维方法，stacking-and-striding 2 consecutive frames or by a stride-2 pooling in the convolution layer，带来了一定的识别率的提升。
* Speed perturbation and SpecAugment
* fairseq toolkit
* adam ,  learning rate linearly warms up from 1e-5 to 1e-3 in the first 8000 iterations and stays at 1e-3 during the rest of training
* ==we mainly use a 12-layer transformer architecture with di = 768: per-head dimension is always 64 and the FFN dimension is always set to 4di.==
* transformers are more prone to over-fitting, require regularization
* 

### Effect of Positional Embedding

* 对比了3种 position embedding的效果
* conv： 2 vgg blocks，这个效果最好，==这个具体的方式，需要探究一下，论文中没有详细讲清楚==

![image-20210607193050018](..\images\image-20210607193050018.png)

### Transformer vs. BLSTM

### Effect of Iterated Loss

* 网络太深训练不起来，加了这个能训练起来了
* 

![image-20210607195721882](..\images\image-20210607195721882.png)

### Limited Right Context

* 对inference时候可以看到的下文做限制，发现如果正确的上下文数量足够，仍然可以参数合理的wer
* 虽然上面可以用，但是更大的下文窗口还是有利的，所以本文还是采用大的下文窗口，所以对下文输入的依赖，导致其不适应流式app

## 总结

* 本篇论文主要在于探索将Transformer应用到ASR中的一些提点的操作，hybrid的，主要探索了position embedding和iter loss, 感觉整体来看，做的事情不多，创新点一般，更多的是方法的应用。

## 引用论文

《A time-restricted self-attention layer for asr》2018

《Self-attention networks for connectionist temporal classification in speech recognition》2019


# Joint CTC-attention based  end-to-end speech recognition using multi-task learning

> author: Suyoun Kim
>
> year: 2017

## Abstract

* However, we observed that the performance of the attention has shown poor results in noisy condition and is hard to learn in the initial training stage with long input sequences，解释原因，主要在于缺乏CTC从左到右的约束，注意力模型过于灵活从而无法预测正确的对齐方式。
* 提出了一个joint ctc-attention的多任务联合训练框架，来一定程度解决对齐问题



## Introduciton

* tradition hybrid approach, DNN-HMM，将系统分解为多个不同的单元，包括，声学模型，语言模型等等等。
* e2e 方法可以纠正因为训练不相交而产生的次优问题。
* e2e方法可以分成两派：CTC和attention-based encoder decoder
* Since the attention model does not use any conditional independence assumption,it has often shown to improve Character Error Rate (CER) than CTC  when no external language model is used , but noise and ==long input sequence hard to learn==, perform bad.  ==加粗是本文解决的关键问题==
* 

## Joint ctc-attention mechanism

* 略

## Attention-based encoder-decoder



![image-20210603165502465](..\images\image-20210603165502465.png)

* $y_{1:u-1}^*$ is the gt of the previsou characters

![image-20210603165646541](..\images\image-20210603165646541.png)

## Joint CTC-attention

* multi task
* 结构如下：

![image-20210603171734496](..\images\image-20210603171734496.png)



* 加入CTC可以增强alignment ， 并且加速网络训练
* ![image-20210603173459316](..\images\image-20210603173459316.png)

## Experiments

* The CHiME-4 corpus was recorded using a tablet device in everyday environments : a cafe, a street junction, public transport, and a pedestrian area
* we used only 32 distinct labels: 26 characters, apostrophe, period, dash, space, noise, and sos/eos tokens. CTC use blank instead of sos/eos
* encoder: 4-layer BLSTM 320cells
* decoder: 1-layer LSTM 320cells

![image-20210603174031868](..\images\image-20210603174031868.png)

* ctc可以加速训练，调大ctc loss的比重，训练效率明显提升。
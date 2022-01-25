# Training data-efficient image transformers& distillation through attention(Deit：Data-efficient image Transformers)

Author: Facebook AI (Hugo Touvron)

github: https://github.com/facebookresearch/deit

---



## Abstract

- 少量训练数据（ImageNet）就可以训练，一台机器就可以训练（8卡可在3天训练完成）
- sota的水平
- teacher-student strategy to transformers: token based distillation,当使用convnet作为teacher效果会更好

## Introduction

- no conv layer
- DeiT-S -> ResNet-50
- a new distillation procedure based on a distillation token
- 采用这种蒸馏方法，convnet相比transformer反而是更好的老师。

## Related work

- class token ： a trainable vector
- vit论文实验显示，在小的分辨率下训练，然后在大的分辨率下finetune能取得更好的效果

## Distillation through attention

- soft distillation
- hard-label distillation: 就是在soft的基础上，取了一个argmax。另外利用distillation得到的伪标签还有一个作用，==就是屏蔽掉在训练中由于augment带来的标签错误的问题。==
- distillation token: 本文创新点：Deit在最后面插入了一个distillation token,然后利用这个token来进行蒸馏loss的计算, 这个token也是learnable的
- inference阶段，利用这两个token的预测结果进行融合。
- ==这个启发倒是可以在我们的项目中试试看，这样的多分支蒸馏的效果还真没==
- ![image-20211021110927626](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211021110927626.png)



## Experiments

* convnet teacher 比 transformer teacher 更好， default teacher RegNetY-16GF
* 下表介绍了不同的蒸馏策略带来的结果：可以看到只使用hard label 和只使用teacher label还是有一定差距的，当然hard+ teacher的效果是最好的，==但是提点好像也并没有很明显。==

![image-20211021113949193](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211021113949193.png)

- ==finetune:== 小数据集测试cifar-10, 重头开训，训练了7200epoch,  rescale to 224x224,取得了不错的效果，不过还是没有直接finetune的结果高。==说明预训练模型还是有效的。==

- Training details:

  > 蒸馏的参数： 温度：3.0 和  比例： 0.1
  >
  > train at 224 and finetune at 3

![image-20211021121904206](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211021121904206.png)

## 总结

​        facebook这个工作是在vit的基础上进行的，主要在于增加了==蒸馏的==训练手段，使得模型的训练时间以及可用的数据量降低了，并且通过蒸馏还有效的提高了模型的效果。


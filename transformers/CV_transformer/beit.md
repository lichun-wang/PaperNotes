# BEiT: Bert Pre-Training of Image Transformers

author: microsoft research

github: https://github.com/microsoft/unilm/tree/master/beit

---

## Abstract

- 仿照bert，提出了一个基于mask的视觉的transformer预训练模型。
- mask后需要预测，不可能预测完整的图像patch的信息，所以需要对图像提取visual token,这里采用的是dall-e中的方法。
- 预训练得到预训练模型后，通过finetune进行下游任务的训练
- 实验结果证实，本文的方法相比于以前的预训练方法，还是挺有竞争力的，结果比vit直接训练好要。
- imagenet-1k:83.2%, large beit, 86.3%



## Introduction

- 视觉没有vocabulary，所以不能直接用softmax进行mask的预测
- masked image model, 简称MIM,如下图，利用image patches以及visual tokens进行预测，visual tokens是通过vae得到的（dall-e那个方法）
- 作者在2个任务做了下游的实验，分别是分类和分割
- 不仅预训练表现好，在收敛速度和稳定性方面也有一定的改善

![image-20211102192100684](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211102192100684.png)

## Methods

### Image Patch

- 224x224图像会分成14x14的patch，每个patch的尺寸是16x16

### Visual Token

- following Ramesh的Dall-e(https://github.com/openai/DALL-E)的工作，dVAE
- Gumbel-softmax relaxation
- vocabulary size == 8192

### Backbone

- follow vit

### MIM

- random mask some percentage of image patches, 40%
- masked patches with a learnable embedding
- blockwise masking , 就是每次mask一块block，我理解就是避免随机带来的太分散了。
- ![image-20211102200325856](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211102200325856.png)



### From the Perspective of Variational Autoencoder(VAE)

- 此处没看懂，后面需要针对该VAE部分进行学习。

### PreTraining Setup

- vit-base的参数设置，
- 16x16的patch提取
- augmentation: random resized crop , horizontal flipping , color jittering
- pretraining : 500k steps, 800epochs, batch_size:2k
- Adam  $\beta_{1}=0.9, \beta_{2}=0.999$
- lr = 1.5e-3, warmup 10 epochs, 
- cosine decay
- weight decay is 0.05
- stochastic depth : rate 0.1
- 16块 V100 32G 训练5天。

### Fine-tuning  beit on downstream vision tasks

#### Image Classification

- 加一个分类层，

- 使用average pooling, ==这里遗留一个问题，就是embedding是怎么做的，然后average是作用在哪里的，需要看代码==，公式是这样的：W是DxC的

  ![image-20211103192452433](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211103192452433.png)

  embedding如下： nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

  average，这里看了代码了解到， 作者设置了两个模式，如果不采用avg模式，就直接使用前面添加的[cls]进行分类，跟vit保持一致了，如果是选择了avg模式，就使用除了[cls]之外的所有,在patch维度取平均后，进行了norm，然后进行分类。代码在下面。

  ```python
  # embedding代码：
  nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
  
  # cls 分类
  if self.fc_norm is not None:
              t = x[:, 1:, :]
              return self.fc_norm(t.mean(1))
          else:
              return x[:, 0]
  ```

  

  - **补充：**BEiT中使用了relative_position_biases 应用到每一个self-attention中，就是在self-attention内部，还有一个位置信息+到了atten的结果上，这个论文中没有体现，代码中写的。

  #### Semantic segmentation

  - follow SETRPUP(Zheng 2020)
  - deconvolution layers

  #### Intermediate fine-tuning

  - self-supervised 后，在imagenet-1k上finetune一下，然后再对downstream task进行finetune

  

  ## Experiments

  - 有一点记录一下，在224尺寸finetune后，在384尺寸继续finetune了10个epoch，结果又提高了1个点+
  - *scaling up to larger size*:  vit384 变成 vit_L 384后，变得worse了，**说明数据不够了**
  - [ ] 下面这个小图画的挺好。
  - [ ] ![image-20211104161538446](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211104161538446.png)

  

  ## 附录

  ![image-20211104111741894](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211104111741894.png)

  

  

  ![image-20211104111818699](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211104111818699.png)

  

![image-20211104111830969](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211104111830969.png)


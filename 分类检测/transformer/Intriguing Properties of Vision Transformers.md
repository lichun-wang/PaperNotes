# Intriguing Properties of Vision Transformers

>

---

auther: Muzammal Naseer

Github: https://git.io/Js15X.

---

## Abstract

- transformers具有较好的鲁棒性（to severe occlusions, perturbations and domain shifts）即使图像被遮挡了80%,识别的准确度依然可以达到60%
- 对于遮挡的鲁棒性并不是由于纹理的偏差，相反我们发现vit相比cnn可以明显的减少局部纹理的偏差.当我们去训练encode形状特征的时候，vit展现了跟人类相当的分类能力，这在文献中是前所未有的。
- 使用vit来编码形状展示，在没有pixel监督信息下，在语义分割也得到了喜人的结果。
- vit model现成的特征可以进行特征融合，可以带来更高的acc。

有一篇工作将bias的可以看一下：《ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness》



## Introduction

- 卷积学习到的是局部的相关性，比如边缘，轮廓信息等；self-attention可以有效的学习全局的相关性，比如较远部分的关系。

- 卷积是内容无关的，因为相同的filter会作用在整个input，而不管他们的距离；而在实验中，我们发现vit可以自由的调整他们的感受野来应付数据中的滋扰以及增强特征的表达。

- 本文进行了系统的实验，发现如下现象：

  > 1. vit对遮挡等，具有较好的鲁棒性，在imagenet val中，即使遮挡80%，依然有60%的准确率。
  > 2. 对于纹理和形状，cnn经常对纹理感兴趣，但vit对形状具有很强的识别性，比如绘画，形状比纹理更重要。
  > 3. 相比cnn，vit在一些扰动因子上具有更好的鲁棒性，比如：patch 打乱，攻击扰动，以及一些自然噪声等。但是与cnn相似，在针对形状的训练中还是比较容易被攻破，对于对抗攻击等
  > 4. imagenet训练的vit，具有较好的迁移性，将vit的现成的层拿出来，就可以较好的适应新的domain，这在比如few-shot learning, fine-grained recognition ,  scene categoriation 或者长尾分布上都有一定的发展。

  另外除了，本文的发现以外，我们也介绍了一些新的design的思想来增强vit的效果。最后我们修改了一个deit结构来增强对于形状特征的学习。

![image-20211216160303135](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211216160303135.png)



## Related Works

- 卷积在独立同分布的 数据上 表现sota，但是对于分布迁移的效果还是比较敏感的，比如攻击噪声 ， domain shifts.

- [15]显示vit对于高频的修改更加的鲁棒

- Stylized ImageNet,一个对imagenet做了风格化的数据集，去除了纹理信息，使得训练的cnn 模型可以更好的学习形状信息。
- shape-focused learning ??咋做的？？

## Intriguing Properties ofVision Transformers

### Are Vision Transformers Robust to Occlusions?

- mask方式：subset of the total image patches , mask的pixel像素值设成0， PatchDrop方法。本文进行了三种掩盖方式，1.随机，2.前景掩盖，3.背景掩盖。

  > 前景掩盖，使用的是dino进行的前景分割

  ![image-20211216164005281](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211216164005281.png)

  

- 效果如下图：看下图就很能说明问题了。

![image-20211216170947306](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211216170947306.png)



### ViT Representations are Robust against Information Loss

- 将特征图可视化后可以发现，越深的layer对于保留的信息的关注度更高。
- ![image-20211216173110696](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211216173110696.png)
- 作者进一步做了相关性试验，验证原始图像与掩盖后的图像的token输出的相关性，分别取了resnet50和vit token之前的的特征图做对比，对比结果如下：可以看到，transformer类模型的相关性明显更高。

- ![image-20211216173122368](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211216173122368.png)

###  Shape vs. Texture: Can Transformer Model Both Characteristics?

​       根据文献[9]，作者尝试了这个方法来做实验，发现vit对于形状的分类能力是高于conv的甚至达到了人类的水平，但是在自然图片上又会导致acc的显著下降。为了解决这个问题，作者引入了一个shape的token到网络中来专门学习shape以达到可以同时关注shape和texture的目的。并且这个shape token使用了一个shape-bias特别strong的conv模型来进行监督。

（1）training without Local Texture

- 在Stylized ImageNet数据集上训练，发现当训练采用大量的Augment的时候，模型波动很大，所以作者去掉了所有的augment
- transformer表现的比较好，甚至达到了人类的水平。

![image-20211217104201198](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217104201198.png)

（2）shape distillation

​        接下来的问题就是如何在自然场景和形状特征都具有较好的准确性呢？这里作者引入了另外一个token来专门学习形状特征，同时用训练较好的卷积模型对这个token做蒸馏，（==这里为啥不用shape-bias更好的vit当teacher呢？==）并且计算了class 和 shape token 以及 class dis_token的cosine相似性，发现蒸馏后可以有更大的相似性，说明我们这种设置多个token的方法还是有益处的，并且通过蒸馏可以同时学到形状很纹理特征，在nature image表现得到提升。

![image-20211217141455239](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217141455239.png)

（3）Shape-biased ViT Offers Automated Object Segmentation

​             不管是去掉纹理信息来训练，还是加入shape蒸馏来训练，vit都专注于前景信息而忽略掉背景信息，这就可以来完成自动的前景分割，尽管并没有用像素级别的数据来进行训练。

![image-20211217142142387](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217142142387.png)





### Does Positional Encoding Preserve the Global Image Context

- 作者将图像的patch的顺序打乱，实验证明，打乱了顺序的图像，vit的效果要比conv的效果要更好一些，这也说明了positional encoding对于图像的正确识别不是至关重要。模型也并没有通过保存在positional encoding的序列信息来恢复全部的图像信息。对于没有加入position encoding的图像，对于permutation的识别是会更好的。
- ![image-20211217150046250](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217150046250.png)
- 当patch size的尺寸发生变化时，permutation不变性与unshuffled的精度一起下降。
- 综上，我们将permutation invariance的表现归功于vit的动态感受野

### Robustness of Vision Transformers to Adversarial and Natural Perturbations

### Effective Off-the-shelf Tokens for Vision Transformer

- 作者拿出了每一层的cls token,来对比其作用，如下：

![image-20211217152431778](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217152431778.png)

- 说明最后的几层的cls token的作用更大，所以作者尝试将最后几层的cls进行融合来取得更好的效果，做法就是transfer learning. 实验发现，融合最后的4个block表现效果最好。同时concat 所有的block和只concat 最后4个的表现相差不大，但是后者的参数量多了很多。有个例外是flower数据集，concat所有的效果要比后4个高1.2%
- 

## Discussion and Conclusions

- 4块 V100
- 

## 附录

1. random patchdrop
   - 当mask的patch是model训练patch的整数倍的时候，vit的表现是比conv要好的，但是当覆盖的过大的时候，conv和vit的表现都会下降，猜测主要原因是移除了太多的视觉信息，使得难度太大了。

2. random patchdrop + offset

   - 按照patch进行mask表现挺好，如果加了offset表现还会好吗？作者进一步做了这个实验，发现整体表现还是可以的的，但是在vit-L下降的很是明显：跟其他模型不同的是vit-l的参数量很大，但为啥就会出现这个现象呢？没搞懂。

     ![image-20211217171650647](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217171650647.png)

3. 如果是pixel drop 回事怎么样的呢？

   - 这次vit-L表现有最好了。。。，但下降趋势确实要比前面按照patch进行mask的要更明显一点。

   ![image-20211217172800492](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211217172800492.png)

4. 对比了swin以及其他的路regnet-Y等模型，最终的结果还是transformer 对patch drop更友好一些。
5. 
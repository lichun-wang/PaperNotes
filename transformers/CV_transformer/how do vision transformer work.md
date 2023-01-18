#  How do vision transformers work

* multi-head self-attention 不仅改善了acc更改善了泛化性，是通过将loss landscape更加的平坦达到的这个目的，**并且这个改善是通过data specificity得到了，而不是long-range dependency得到的**。
* **vit也是一个非凸的loss**，大的数据及以及loss landscape smoothing方法减缓了这个问题。
* MSA和Conv表现出来相反的表现，**MSA是低筒滤波器，Conv是高通滤波器**
* 每个stage的最后的MSA在预测而中扮演了很重要的角色
* 提出AlterNet,在Conv的网络中，将每个stage最后的block替换成MSA,AlterNet比CNN不仅在大数据表现好，小数据集也表现好。

---

## Related work

- transformer没有归纳偏置，但没有约束真的好吗？不一定，swin等这些模型的表现也说明了约束在一定程度上能提高模型的表现。

- transformer一些有趣的表现：

  > 1. MSAs 比CNN表现好
  > 2. MSAs 比 CNN 更鲁邦
  > 3. MSA 后面的layer可以显著提高预测性能

- 作者认为，MSAs工作的主要点在于，一个可训练的空间平滑（这个从self-attention的公式就可以看出来，就是做了特征平滑），这种平滑不仅改善了acc以及鲁棒性，同时平滑了loss landscapes

- MSAs negative Hessian eigenvalues在小数据集内，说明小数据集loss是非凸的，会扰乱训练，当数据增大之后，会缓解这个问题。

- resnet的loss landscape更陡峭，vit transformer更扁平，极坐标可以看到，transformer是比较平滑的收敛，但是resnet就是比较乱的，同时hession，vit开始存在负值，说明非凸，影响训练。

![image-20220321173559195](..\..\images\image-20220321173559195.png)



![image-20220322161600127](..\..\images\image-20220322161600127.png)



* 是什么导致的vit在小数据集上效果不是很好呢》

  > 1. 是过拟合了吗？通训练来看，应该不是，因为train和val依旧是正相关的
  > 2. 其实是非凸的loss影响了训练，而当数据量比较大的时候，可以消除负的hession极值点，进而表现更好了。
  > 3. Loss landscape smoothing methods aids in ViT training.  作者利用GAP替换CLS token来进行分类，效果变好了。
  > 4. MSAs flatten the loss landscape.
  > 5. A key feature of MSAs is data specificity (not long-range dependency).



![image-20220322101430111](..\..\images\image-20220322101430111.png)

![image-20220322101351636](..\..\images\image-20220322101351636.png)

* **作者通过向训练中增加噪声，验证了，conv是高频滤波器，MSA是低频滤波器，所以conv对纹理比较在意，而MSA对形状比较在意。**

- MSAs aggregate feature maps, but Convs do not
-  MSAs closer to the end of a stage to significantly improve the performance



* 基于上面的分析，设计了alter-resnet-50网络，对小数据集效果更好的网络。作者是从后往前，一点一点替换网络conv为self-attention，一点点提升效果。

![image-20220322113614574](..\..\images\image-20220322113614574.png)

![image-20220322162203599](..\..\images\image-20220322162203599.png)

---

## 背景知识：论文：Visualizing the Loss Landscape of Neural Nets

这篇文章主要介绍了loss landscape是如何画出来的，以及设计的思想。

关于画出的二维的图像，主要的公式就是下面这个,解释一下就是，先根据模型的参数$\theta$以及两个随机的方向$\alpha,\beta$,通过一定的步长来调整模型，然后计算loss画出对应的曲线。

$f(\alpha,\beta)=L(\theta^*+\alpha\delta + \beta\delta)$

这里有一个点需要提一下的是，filter-wise normalization，简单来说就是两个随机方向的大小和模型参数要保持一致，为什么要这么干呢，举个简单的例子，如果前一层的参数扩大10倍，后一层的参数缩小10倍，得到的结果是不变的。如果发生这种情况，画出来的loss就体现不出来了，所以作者统一做了一个归一化，保证尺度的一致性。
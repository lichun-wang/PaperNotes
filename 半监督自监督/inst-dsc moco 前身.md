# Unsupervised Feature Learning via Non-Parametric Instance Discrimination

>本文是一篇自监督学习算法，主要思想在于，作者发现，在在有监督的图像识别过程中，一张图像，在相似类别的得分会相对比较高，而不相似类别会相对比较低，这说明，在没有这类标签的情况下，模型可以get到类别之间的相似性，那么如果我们把类别细化到极致，把每个图像都当成一个类别，是不是可以利用图像的相似性来实现图像的判断呢？本文将图像通过特征提取了128维的向量，然后利用点积来计算每张图的相似性，完成训练。在测试阶段，通过non-param 的 knn算法，实现图像的分类。

---

- Github : http://github.com/zhirongw/lemniscate.pytorch.

---

## Abstract


- 作者发现，即使是用有标签的数据来进行训练，对不同的类别，模型也能捕获其中的视觉的相似性。那么我们能不能通过特征进行实例判别，来学习一个好的特征表达来捕获这种实例之间的相似性，进而来替代类别？
- 我们把这个灵感叫做实例间无参数分类问题，然后使用noise-contrastive estimation来解决计算复杂的问题
- 实验证明，这个方法是不错的，大幅度超过了无监督的imagenet的sota模型，**并且在增加数据集以及使用更大的模型，可以得到显著的持续的提升。**
- 我们的模型很小，如果每张图像提取128dim的features, 我们只需要600MB就可以存储1百万图像，这样使用knn算法就可以快速的找到相邻的图像，进而实现分类。

## Introduction

- 在分类模型中，top5的正确率会显著的低于top1的正确率
- 在分类的过程中，相似的图像会得到比较高的分数，说明这种典型的判别学习方法可以自动的发现明显的相似性，换句话说就是，明显的相似性不是通过标注学到的，是通过数据自己学到的。那既然如此，我们是否可以将类别推到极限，把每个图像都当成一个类别？通过这样的方式，我们能学习到一个很好的实例间的相似性吗？
- 既然，通过class-wise的方式可以学习到class内相似的特征表示，那么如果我想要判断每个实例的相似性，那是不是就得把每个实例当成一个类别来学习，即instance-level
- 但是当每个图像都是一个类别的时候，会出现一个问题，imagenet会有1.2million个类别。这个时候用softmax已经不行了
- 为了更好的评估unsupervised learning的效果，过去的工作一般使用一个线性的分类器，比如svm,利用学到的特征进行分类，但是学到的特征真的是线性可分离的吗？这个还不好说，所以本文没有使用这样的方法进行测试。
- 本文将instance-level discrimination 当成一个metric learning问题，通过non-parametric的方式来计算相似性，就是说，把instance都存在memory bank里面，而不是存在网络参数里，本文采用kNN的方式进行测试，每个图像保存128dim的特征。来规避上面的问题

## Related Works

- Generative Models

  > GAN
  >
  > RBMs (whats this)
  >
  > Autoencoders

* Self supervised learning

  >预测任务： predicting the context [2], counting the objects [28], filling in missing parts of an image [31], recovering colors from grayscale images [47], or even solving a jigsaw puzzle [27]. For videos, self-supervision strategies include: leveraging temporal continuity via tracking [44, 45],predicting future [42], or preserving the equivariance of egomotion [13, 50, 30].
  >
  >也可以多种组合，打出组合拳
  >
  >然后值得思考的是，self-supervised learning 可以捕获实例间各部分的关系，但是不清楚的是，这样的任务对语义识别是有帮助的吗？以及，哪个任务是最优的呢？


- Metric learning
- Exemplar CNN (跟这篇论文一个比较相似的工作，NIPS 2014)
- 

## Approach

![image-20220105144459064](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105144459064.png)

### Non-Parametric Softmax Classifier

如果是参数的分类器，那么公示是这样的：这里的w可以理解为fc层的参数，这里的问题是，这样拿到的类别的概率，其实不完全是实例间的比较，因为乘了w相当于做了投影。

![image-20220105144912252](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105144912252.png)

那无参数的分类器是什么样的呢？

![image-20220105145058830](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105145058830.png)

简单来说，就是v要跟所有的向量做内积，算相似度，然后 ||v|| = 1, 通过L2-normalization做的归一化， $$\tau$$是温度系数，所以目标就是最大化i=0-n的概率的乘积，或者最小化-log，即loss如下所示：

![image-20220105145705425](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105145705425.png)

### Learning with a memory bank

对于所有的图像，这里保存了每张图像最后提取的特征在V中，方便上面计算相似性，初始化的时候V中是随机向量，然后在每个iter的时候，对相应的图像进行更新。

如果采用带参数的学习，比如加上fc,那么对于类别是比较敏感的，缺乏泛化性，我们这种方式，对于新类别就很友好

### Noise-Contrastive Estimation

如何解决几百万的计算问题呢？这里提到了几个方法

- hierarchical softmax:  采用一个huffman树（二叉树），然后通过路径来计算每个节点的概率。这里可以将计算量从V变成logV
- noise-contrastive estimation(NCE)
- negative sampling

网上看到的关于这方面的介绍文章：https://ruder.io/word-embeddings-softmax/

这里采用的是NCE算法

- nce算起来还是挺复杂的，文章里介绍的挺数学的，总之，最后的loss变成了下面这样，后面的对vi做的平方项是一个正则，就是由于bank更新的太慢了，所以这里加了前后两个bank的diff作为loss的一部分。

  ![image-20220105160659698](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105160659698.png)
  
- 可以看到，加入了后面的正则项后，训练变得更稳定了。
  
  ![image-20220106150440850](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220106150440850.png)

### Weighted k-Nearest Neighbor Classifier

- 在统计knn的时候，会根据相似度对不同的样本分配不同的权重，权重的计算方式为$\exp(s_i/\tau)$

- 然后每个类别的得分就是如下这么算的：k = 200

  

![image-20220105161154282](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105161154282.png)

## Exp

####  一、首先对比有参数和无参数的实验效果：

采用cifar-10数据集，这个数据集数据量比较少，可以计算公式（2）。

实验对比了训练SVM和knn的实验效果：如下

![image-20220105164205419](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105164205419.png)

#### 二、图像分类

- 前几列是采用SVM进行的测试

![image-20220105174000246](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105174000246.png)

* 1.28M的图像，每张提取128dim，消耗600M,搜索时间大概20ms Titan X GPU

- 特征的泛化性： 在imagenet上训练，然后在Place上进行测试，测试结果如下：
- ![image-20220105175112978](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105175112978.png)

- 128dim挺好，256dim下降了
- ![image-20220105183033749](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105183033749.png)
- 

### semi supervised

将imagenet的一部分标签拿出来，其余的部分作为无标签处理，来验证semi supervised的效果，如下：

![image-20220105183357358](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220105183357358.png)
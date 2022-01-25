# CvT

> 本文作为将conv应用到transformer里的一篇文章，主要有如下几个创新点，1：采用了分层的结构，2，提出了convtional token embedding，进而去掉了position embedding，3，采用conv替换了计算QKV的Linear，并且采用depth-wise以及point-wise,节省计算量，没有提取patch，是基于像素来进行的。







Follow ViT



![image-20211108191416776](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108191416776.png)

* convolutional token embedding: 利用卷积实现token的映射，从H*W*C -> H‘W'C' -> { (H’W') C' + normalize }, 没有基于patch？是的
* 在求QKV的时候，没有使用linear映射，使用的是depth-wise的conv实现。Depth-wise Conv2d →
  BatchNorm2d → Point-wise Conv2d，节省时间手段1。
* 如下图c，在conv projection的时候，对key和value进行了缩小操作，stride=2，来节省时间手段2。

![image-20211108193204099](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108193204099.png)



## Discussions

- **removing positional embeddings**, 因为使用了卷积的embedding，使网络具备了空间关系的能力，所以可以去掉positional embedding,==这里有个疑问，真的能替代吗？感觉self-attention内部没有带上位置信息呀==
- 

![image-20211108195241405](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211108195241405.png)
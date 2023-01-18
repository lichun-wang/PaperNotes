# SLIP: Self-supervision meets Language-Image Pre-training

> 这篇文章很简单，思想也很简单，主要就是将图像文本对的学习以及 图像图像的对比学习进行了融合。

---

Facebook AI Research

GitHub :  https://github.com/facebookresearch/SLIP

---



​        思想就是，我们在图像做自监督对比学习的时候，往往没有利用亿级别的数据，因为没有证据表明对比学习是scale well的，这里从moco也可以看出，多了100倍的数据，但是效果却并没有得到多少提升。但是clip的训练是需要大量数据的，那么，图像对比学习与图像文本对的对比学习，是否是相互互补的呢？

![image-20220315164956796](..\images\image-20220315164956796.png)

![image-20220315202438591](..\images\image-20220315202438591.png)

### 确实是有促进作用的

![image-20220315194512918](..\images\image-20220315194512918.png)

### finetune的时候，模型scale影响不是特别的大，当然也会提一些，linear probing的时候，往往会有更大的提升。



### 如果采用对比学习的模型作为clip的初始化参数，会怎么呢？ 答案是：效果没有联合训练好。





### 如果slip的图像-文本对比学习、图像图像对比学习，采用不一样的数据集，会产生怎么样的效果呢？ 答案是： 结果跟采用相同的图像效果差不多的。
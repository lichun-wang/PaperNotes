# SwAV

> 这篇文章的大致思想是，将instance discramination的方法进行变种，不再将每个图像当做一个类别进行softmax对比，而是将图像进行聚类，通过对比相同图像在不同augment情况下载聚类C上的映射，来进行判断。具体来说，就是对qt和qs进行相互的预测，相当于通过一个聚类的方法，将类别数量降到了C的维度上面，论文默认是3000。
>
> 另外一个比较有用的技术是 multi-crop,提点效果比较好
>
> 论文中有一些公式我并没有看懂，比如，如何设计loss使得在进行C投影的时候，可以有已经的区分度。后续准备看看deep cluster v1+v2来进一步学习一下这系列的方法。



![image-20220118200931484](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220118200931484.png)



* loss function:

  ![image-20220119150433941](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20220119150433941.png)

## Result

### Training with small batches

- bs = 256 , 也取得了不错的效果。

- swav 相比 moco v2 ，只需要存储更少的特征即可
- 在训练上，SwAV比moco v2 和simclr要慢一点，但是swAV需要的epoch数量少（1/4）就超过了mocov2
- 


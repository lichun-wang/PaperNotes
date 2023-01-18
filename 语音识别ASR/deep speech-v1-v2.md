# Deep Speech: Scaling up end-to-end speech recognition

**Baidu 2014**

​         比较 早的一篇文章，来自Baidu，主要采用了RNN作为声学模型，结合语言模型n-gram进行CTC解码，取得了不错的效果。

​          在深度学习之前的语音识别系统，大量的依赖语言学者对特征提取的设计，本文利用深度学习自主学习的特点，利用大量的训练数据，学习出特征，并利用ctc实现了语音和标注文本的对其工作，实现了端到端的ASR.

训练的时候，为了提高抗噪声能力，利用噪声合成的方式，生成伪造数据，提高噪声环境的效果，噪声合成的方式包括：电视、洗碗、咖啡厅、餐厅、下雨声等等。

​          训练语料：

<img src="..\images\image-20210528102534607.png" alt="image-20210528102534607" style="zoom:50%;" />

​           loss: CTC Loss          

​           语言模型解码过程，beam search：

​                                                   ![image-20210528102633933](..\images\image-20210528102633933.png)

​           网络结构图： 前三层是单向RNN，第四层是双向RNN，第五层接受双输入：

<img src="..\images\image-20210528103652618.png" alt="image-20210528103652618" style="zoom:50%;" />


# Deep Speech V2

v2是在v1上进行的升级，采用了更大的语料，重新设计了网络结构，引入了CNN在网络头部，一定程度减少了计算量。

![image-20210601103914023](..\images\image-20210601103914023.png)

引入了Sequence-wise的normalization，但是产生的模型提升效果不明显。

因为英文字符比较多，导致了加大stride的时候会出现一些问题，作者设计了bigrams的策略，即[th，e, space, ca,t,space,sa,t] 这样的方式。

Language Model还是采用的原来的方式。

本文介绍了很多的GPU上面加速的方法，包括数据并行化，梯度，还有ctc-loss在gpu上的实现，内存malloc的管理等。


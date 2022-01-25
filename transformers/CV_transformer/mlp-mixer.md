# mlp-mixer

Auther: google Brain

---



## Abstract

- MLP-Mixer含有两种layer: 一种是独立作用于每一个patch，混合location feature,另一种作用于patch之间的，混合spatial information

## Introduction

![image-20211020144322482](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211020144322482.png)

## Arch

- 首先提取patch，并且对patch做维度的映射 得到输入 X: [S, C]
- token-mixing MLP： 作用于S,  (简单来看就是对$X^{T}$进行操作)
- channel-mixing MLP : 作用于C
- 可以写成下面的公式，每个MLP含有两个Linear：
- ![image-20211020150844011](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211020150844011.png)
- ![image-20211020150936854](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211020150936854.png)

- channel-mixing的过程中，由于是使用的fc，所以channel的增长会带来参数量成倍的增长，所以为了控制其增长，这里channel的增长是缓慢的，但是实验证明这并没有影响模型的效果。
- 不像resnet的金字塔结构，MLP的特征图尺寸没有变化
- 也使用了skip-connection and Layer Norm
- Mixer does ==not use position embeddings== because the token-mixing MLPs are sensitive to the order of the input tokens, andtherefore may learn to represent location
- 


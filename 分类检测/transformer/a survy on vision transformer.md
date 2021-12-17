# A survey on vision transformer

>

---

Author:中国科学院大学，东南大学，lenovo ai lab

一个解读：https://mp.weixin.qq.com/s/_th7rXfZDuSu2xo7gdPp0w

---

- Because of the restricted feature subspace, the modeling capability of a single-head attention block is coarse

## Classification

### 1.Original Visual Transformer

- Vit -> patches; JFT-300M

### 2.Transformer Enhanced CNN

- VTs
- BoTNet

### 3.CNN Enhanced Transformer

- CNN的inductive bias 是局部，变换不变性，但是这给cnn一个上界，即使有充足的数据。
- DeiT:  conv teacher  , 蒸馏
- ConVit
- CeiT and LocalVit
- CoAtNet
- ...

### 4.Local Attention Enhanced Transformer

- TNT
- Swin Transformer
- ==**VOLO**==

### 5. Hierarchical Transformer

- T2T-Vit
- CVT

### 6.Deep Transformer

- CaiT
- Refiner

### 7.Transformers with Self-Supervised Learning

- MoCo v3
- BEiT

### 8.discuss

- DeiT的训练模式还是被普遍使用的。
- 对于局部的感知是不可或缺的，这点在VOLo和swin中得以反映。
- 浅层使用conv是有益的，可以帮助模型在浅层得到更好的局部特征。
- 未来可能会有更多的conv和transformer的融合模型出来。



![image-20211117191002444](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211117191002444.png)



![image-20211117191617139](C:\Users\wanglichun\Desktop\Typera\TyporaPapers\images\image-20211117191617139.png)

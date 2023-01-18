# Trustworthy AI

>京东AI Research

---

## Abstract

目前的AI系统存在一些问题，比如：面对攻击的脆弱性，偏见，缺乏隐私，这些都会降低社会对ai的信任

可信AI应该包括：鲁棒性，泛化性、可解释性、透明性、可复现性、公平性、隐私保护、符合人类价值观、以及责任

## Introduction

## AI TRUSTWORTHINESS：BEYOND PREDICTION ACCURACY

### Robustness

#### Distributional shift

要考虑多场景多样的数据分布情况，比如，自动驾驶，既要考虑晴天，也要考虑雨天的情况。

#### Adversarial attacks

人脸带个眼镜啥的就可以骗过识别系统

#### Illegal input

打印的面具作为输入

意想不到的输入，比如图像超级大、超级小、自动驾驶里面其他卡车的激光等



The objective of defense can be either proactive or reactive [254]. Proactive defense attempts to optimize the AI system to be more robust to various inputs, while reactive defense aims at detecting the potential security issues such as changing distribution or adversarial samples.

测试鲁棒性 在避免这些问题中 也显得尤为重要。

#### Robustness test

monkey test

performance test

minimal adversarial perturbation 



### Generalization

没有见过的数据的能力

cross-validation、regularization、data augmentation

目前ai对于标注的依赖还是比价大，提高泛化性可以减少数据的依赖。

domain adaption

few shots

pre-training and meta-learning 提高迁移能力。

加入对抗样本进行训练，可能会提高鲁棒性，但是也很可能会降低泛化性。

cross-validation

Rademacher complexity

### Explainability and Transparency

#### Explainablility

每个字节，每个参数的意义，可以帮助 责任的划分

模型设计就考虑 模型的可解释性

事后分析：从输入、中间结果、输出进行模型的分析，

测试：  subjective human evaluation，人类和ai的协作表现评价。

#### Transparency 透明度

设计目的，数据来源，硬件信息，配置，工作条件，系统表现等等。

### Reproducibility 可复现性 

目前顶会基本都会复现代码

全周期的实验记录

reproducibility checklists

### Fairness

性别，种族等。

比如从50个男的和50个女的里面招聘

### Privacy protection

### Value Alignment 价值观的对其

deepfake fraud

产品设计上，道德准则需要考虑进来，比如视频评论系统，不能带有对特定目标人群的攻击言论

技术的发展；将人类价值观强加给机器， 比如 apprenticeship learning 爱你的 evolutionary strategies



## Recent Practices



**人脸识别**： 

==恶意攻击==。隐私泄漏

恶意攻击，用一张照片就可以骗过了 ---- 活体检测 --- deepfake ----- 合成图像检测。

人脸对抗攻击 -- 眼镜

现代的人脸检测对于domain是比较敏感的，不同的采集设别，不同的人种都需要重新采集图像， domain adaption 技术

公平性问题；要考虑到  种族，年龄，性别，效果上要保持平衡

隐私保护问题：受到攻击时 会泄漏人脸数据



**自动驾驶**

鲁棒性： 自动驾驶汽车 会安排多个传感器。各种场景，各种模拟 是 需要的。

透明性和可解释性： 规定L1-L5，DNN的可解释性就差一些



自然语言处理

安全性的需求就会弱一些，但是价值观、道德层面的东西就强一些。

比如：性别歧视： He is a docker 。 She is a nurse， 这种情况下，google提供了 可替换性别的翻译，来减轻这个问题。

比如 对话机器人，可以增加一些标记 来识别用户的风险。

比如 语音助手， 默认是女性的声音，容易让人将其与现实中的女性联系起来，存在一定的偏见，特别对新时代成长的孩子会产生一定的影响，现在也提供声音选择了。

### Recent Multidisciplinary Progresses

要设计与参与者协同 构建可信AI

工业实践

CodaLab ， Development platforms such as CodaLab provide workspaces for reproducible AI experimentation and development

Amazon SageMaker, IBM OpenScale.

不同的部门，不同的人员，对于可信AI的理解是不一样的，所以这就需要有一套标准和准则。

Representatives of these guidelines include the EU Ethics Guidelines for Trustworthy AI [15]

AI industry, such as GDPR, the Chinese Cybersecurity Law, and the Personal Information Security Specification



## A systematic approach

![image-20220421174440676](/Users/wanglichun/Library/Application Support/typora-user-images/image-20220421174440676.png)

Data

数据的bias  =》 debias sampling and debias annotation， 选择标注人很重要。比如，他们要知道标注的内容比如 方言这些。这步不要引入标注偏见。

Data Provenance， 数据起源，意思就是，数据要可追溯，这样才能复现。

数据预处理： 可以删除那些，可能会影响模型表现，或者泄漏隐私的数据。

Anomaly Detection 异常检测（outlier detection），其实就是数据清洗

比如：银行对于假数据比较敏感，会破坏鲁棒性



Data Anonymization 数据匿名化：

Differential privacy 查分隐私

**算法设计**

对抗攻击的鲁棒性， 在训练数据中加入攻击样本。

网络或者正则化 可以克服dnn在攻击中的漏洞，一个动机就是，加入小的扰动并不能改变网络的结果，比如，加一些正则项。比如找到分类边界的下界。

毒药攻击，在训练数据做手脚，可以从数据中解决，或者砍掉对应的神经元



可解释的ML

Attention， CAM，

模型泛化能力

classic：

​           early stopping [420], batch normalization [188], dropout [351], data augmentation, and weight decay

domain generalization

pretrained + finetune

unsupervised pre-training Moco 等等。few shot

算法的公平性

预处理：

> adusting sample importance： resampling   reweighting
>
> adusting feature importance：
>
> data augmnetation

训练中

> adusting sample importance:     adversarial learning

训练后

> 后处理
>
> reweighting predictions of multiple models

隐私计算

SMPC

Federated Learning

发展

异常检测、攻击检测

人类和AI的交互，人工干预



Management

- documentation
- 
































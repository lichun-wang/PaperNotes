# 大纲：

1. 可信AI的背景和重要性

2. 可信AI的概念和具体含义

3. 构建更加可信的内容安全AI系统的探索与实践

4. 可信AI的未来发展与思考

# 内容：

## 一、背景和重要意义

​        近十年来，随着深度学习技术的发展，AI得到了跨越式发展，各种AI产品层出不穷，应用到了生活的方方面面，AI带给我们很多的便利，改变了我们的生活，但许多AI带来的问题也随之浮现：比如：对攻击的脆弱性，对特殊群体的偏见，缺少隐私保护等等。

​        举几个例子：

> 人脸识别的应用广泛，但是人们由于担心隐私泄漏人脸信息。deepfake的攻击，让人觉得人脸不安全。
>
> 男女偏见带来的不公平性： 比如ASR中： he is a doctor, she is a nurse.
>
> AI事故的责任认定：比如：自动驾驶出事了，谁负责？

所以，解决这些问题，提供人们更可信的AI势在必行。

## 二、概念和含义

- 鲁棒性

  robustness refers to the ability of an algorithm or system to deal with execution errors, erroneous inputs, or unseen data

- 泛化性

  AI systems to make accurate predictions on realistic unseen data, preferably even from domains or distributions they are not trained on

- 可解释性、透明性

  ​    understanding how an AI model makes its decision

​                   the lifecycle information of an AI system

- 可复现性

  In terms of AI trustworthiness, this verification facilitates the detection, analysis, and mitigation of potential risks in an AI system, such as vulnerability on specific inputs or unintended bias.

- 公平性

  fair

- 隐私保护

  Privacy protection mainly refers to protecting against unauthorized use of the data that can be used to directly or indirectly identify a person or household.

- 符合人类价值观

  AI systems should be designed to enhance human society’s welfare without violating principles of human values.

## 三、探索与实践

鲁棒性：

> 对抗攻击,  deepfake
>
> 训练中加入对抗样本
>
> 对分类边界的下界进行分析
>
> 易盾举办的 “攻击对抗” 活动

泛化性

> Augmentations，正则化，weight decay等方法，防止过拟合。
>
> 迁移学习，半监督，自监督，moco，clip等尝试 提高模型泛化能力。
>
> 交叉验证。

可解释性

> attention
>
> CAM等

透明性

可复现性

> ​	全周期的完善实验记录

公平性

> 数据采样来平衡
>
> 标注的准确性，避免引入人为偏见

隐私保护

> 涉及隐私数据放云端

符合人类价值观以及责任

## 四、未来发展


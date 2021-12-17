在cv transformer中， dropPath是被经常使用的，就是随机深度 stochastic depth

一般会使用timm库， from timm.models.layers import DropPath



```
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
```

这里你可能会奇怪，最后那行为什么div了keep_prob？这里其实主要原因是考虑的数据分布问题，因为过滤掉了一部分数据分布就变了，所以好除一下。摘录一下别人的解释。

> 假设一个神经元的输出激活值为a，在不使用dropout的情况下，其输出期望值为a，如果使用了dropout，神经元就可能有保留和关闭两种状态，把它看作一个离散型随机变量，它就符合概率论中的0-1分布，其输出激活值的期望变为 p*a+(1-p)*0=pa，此时若要保持期望和不使用dropout时一致，就要除以p。


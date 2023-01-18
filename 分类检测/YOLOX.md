- Auther:  Zheng Ge
- Github： https://github.com/Megvii-BaseDetection/YOLOX.
- 作者解读： https://mp.weixin.qq.com/s/p4Porn9KayizQiQIzTTFKA

## Abstract

- 引入anchor-free
- decoupled head ,simOTA
- Performance:  yolo-nano:  1.8% ap； yolov3 : 3%; YOLOv5-L:1.8% ; 
- provide tensorrt onnx version

##  Introduction

- 近两年目标检测的研究热点： anchor-free； advanced label assignment strategies[37,36,12,41,22,4]; NMS-free detectors[2,32,39]
- YOLOv3 as default
- we boost the YOLOv3 to 47.3%AP (YOLOX-DarkNet53) on COCO with 640 × 640 resolution, surpassing the current best practice of YOLOv3(44.3% AP, ultralytics version2) by a large margin.'
- YOLOv5 640x640； 50.0% AP, supass 1.8%AP
- 

## YOLOX

### Implementation details

- 300 epoch , 5 epoch warmup
- lr = init_lr * (batchsize/64),    init_lr = 0.01,   cos lr schedule,  (8-gpu, batchsize:128)
- input size : 448 to 832 , 32 strides

### YOLOv3 baseline

- DarkNet53 +  SPPlayer
- ==adding EMA weights updating, cosine lr schedule, IoU loss and IoU-aware branch.  We use BCE Loss for training cls and obj branch, and IoU Loss for training reg branch.==

### Decoupled head



![image-20210918121226730](..\images\image-20210918121226730.png)

![image-20210918121247956](..\images\image-20210918121247956.png)

### Strong data augmentzation

- Mosaic and Mixup, and closed it for the last 15 epoches
- After using strong data augmentation, ImageNet pre-trained is no more beneficial , == train all the following models from scratch.==

### Anchor-free

### Multi positives

- not only the center point as positive ; center 3x3 area as positives ; as 'center sampling' in Fcos, balance the positive/negative samplings.

### SimOTA：

- ==Zheng Ge, Songtao Liu, Zeming Li, Osamu Yoshie, and Jian Sun. Ota: Optimal transport assignment for object detection. In CVPR, 2021.==
- simOTA是在上面基础上的优化，去掉了Sinkhorn-Knopp  算法，选择topk替代。
- OTA的主要作用是，对训练的样本的动态分配，可以在训练的时候，自动计算正样本；可参考博客：https://zhuanlan.zhihu.com/p/394392992

### aug的提升效果：

![image-20210922111546338](..\images\image-20210922111546338.png)

  ## Other backbone

### vs yolov5

- use cspnet , silu activation , pan head 

![image-20210922114054103](..\images\image-20210922114054103.png)

### vs tiny, nano detector

![image-20210922114234135](..\images\image-20210922114234135.png)

### model size and data augmentation

- 对于小模型，需要减少augment(mosaic,mixup等)；对于大模型需要增强augment

## 总结

- 本文依然是在yolov3的基础上进行一系列的魔改，增加了anchor free；decouple head; augments等，最终提升了模型的效果，在大模型和小模型中，都取得了一定的优势。

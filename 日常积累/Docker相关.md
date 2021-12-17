## FROM  base

* nvidia/cuda:10.1-devel-ubuntu18.04
* vistart/cuda:10.2-cudnn8-tensorrt7-devel-ubuntu16.04 
* nvcr.io/nvidia/pytorch:0.11-py3
* nvcr.io/nvidia/tensorrt:20.11-py3
* nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 (这个可能会pull不下来)
* nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04



## nvidia提供的cuda的docker下载链接：

- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
- （这里面是没有装python和pip等东西的，都需要自己装）

nvidia 提供的 tensorrt 的docker 下载链接：

- https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-05.html#rel_21-05
- nvcr.io/nvidia/tensorrt:xxxx




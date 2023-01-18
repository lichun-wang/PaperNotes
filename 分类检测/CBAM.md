## Absract

- channel and spatial
- Residual  attention network
- senet



## Channel Atten

* W1和W0共享, to save parameters

![image-20210927114232108](..\images\image-20210927114232108.png)



## Spatial Atten

*  we first apply average-pooling and max-pooling operations ==along the channel axis== and concatenate them to generate an efficient feature descriptor, then add 7x7 conv , and add sigmoid function .

![image-20210927141800944](..\images\image-20210927141800944.png)



- 实验表明， 针对spatial， 利用avg+max进行压缩效果更好

![image-20210927155604533](..\images\image-20210927155604533.png)

- 实验表明，先channel 后 spatial 效果更好

![image-20210927155710351](..\images\image-20210927155710351.png)


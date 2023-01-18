* import cv2
* 修改颜色空间： img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
* 灰度化： img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
* 截取roi区域： roiImage = srcImg[20:50,40:60]
* 二值化： retval，img = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
* cv2.line(img,(0,0),(511,511),255,5)
* cv2.rectangle(img,(20,20),(411,411),(55,255,155),5)
* img = cv2.resize(img,(宽，高))
* img = cv2.resize(img, (0,0), fx=scale1, fy=scale2)
* w, h, c = img.shape # c is channel

* python cv2 resize错误：cv2.error: ... error: (-215) func != 0 in function cv::hal::resize
> 像素值数据类型错误导致了这类错误，作类型转换即可解决： img2 = img1.astype(np.float32)

* 新建图像 new_image = np.full((h,w,3), background, dtype=np.uint8)

* 有一种图像使用opencv读取会出现Premature end of JPEG file, 使用Mxnet读取后会直接报错的，但这个问题使用try except还无法直接捕获，处理还会报错，很烦，最后直接在整个程序加了一个try except解决，看来加try catch很重要哇。

- opencv 转pil

```
import cv2  
from PIL import Image  
import numpy  

img = cv2.imread("plane.jpg")  
cv2.imshow("OpenCV",img)  
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
image.show()  
cv2.waitKey()
```

- pil 转opencv

```
import cv2  
from PIL import Image  
import numpy  

image = Image.open("plane.jpg")  
image.show()  
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)  
cv2.imshow("OpenCV",img)  
cv2.waitKey()  
```

- 读取像素值，注意（i,j）跟opencv是相反的，cv2_2_pil_resize.getpixel((12,50))
- pil resize :  cv2_2_pil_resize = cv2_2_pil.resize((size, size))





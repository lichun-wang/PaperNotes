## gpu高性能cuda实战pdf



如何编译cuda程序

```
nvcc -o main.out main.cu foo.cu
```

cuda 函数前面会加上  \__global__ 这会告诉编译器，这个函数需要用设备编译器编译，即CUDA，其他的在主机编译，即C、 C++

```
cudaError_t cudaMalloc(void**. ptr, size_t size);
```

不能在主机代码中，操作cudaMalloc分配的内存指针

(Void**)&ptr 和 &ptr是等价的，这个如何理解？？



tid = threadIdx.x + blockIdx.x * blockDim.x

\__constant\__ 常量内存读取可以节省内存带宽

cudaMemcpyToSymbol()会复制到常量内存



cuda事件 API ， cuda事件本质是一个GPU时间戳。

cudaEventSynchronize(). cuda事件同步



纹理内存，**tex1Dfetch 来拿到数据**， 后面可以详细看



渲染那章节，没有细看。



cudaHostAlloc(); 固定内存，如果不设置pinned_memory, cpu到gpu的复制过程将执行两遍。先复制到临时的页锁定内存，然后再复制到gpu上，会受限于总线速度。PCIE。



Stream: **有序的工作队列**

cudaMemcpyAsync() , 表示 在流 中 执行一次 内存复制操作。通过参数stream来指定。并且要求，host_a必须是页锁定内存。

```c++
cudaMemcpyAsync(dev_a,  host_a,  N * sizeof(int),  cudaMemcpyHostToDevice,  stream )
```

cudaStreamSynchronize(stream)  同步

零拷贝内存 cudaHostAllocMapped ，不需要内存copy

Cudathreadsynchronibe()将cpu与gpu同步。



多gpu的情况，每个gpu需要一个单独的线程管理，利用cudaSet Device()来指定gpuid 

cudaSetDevice(data->deviceId)



cufft ： Fast Fourier Transform库， 快速傅立叶变换库。

cublas： Basic Linear Algebra Subprograms  线性代数函数库。

cuda-gdb

cuda-Memcheck

visual profiler 工具，cuda zone网站下载



《Programming Massively Parallel Processors： A Hands-on Approach》
















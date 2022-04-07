# TensorRT Plugin接口介绍
插件接口的继承关系
![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/plugin/plugin.jpg)
## nvinfer1::IPluginV2 
用于用户自定义层的插件类，最古老的接口，不支持动态尺寸，有着下列方法

### nvinfer1::IPluginV2::clone() const
克隆插件对象，复制插件的内部参数并返回带有这些参数的新插件对象。
将这个plugin对象克隆一份给TensorRT的builder、network或者engine.

### nvinfer1::IPluginV2::configureWithFormat()
配置这个插件op，判断输入和输出数据类型，维度是否正确。官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型

### nvinfer1::IPluginV2::destroy()
当network，builder或者引擎销毁时调用并销毁自身。

### nvinfer1::IPluginV2::enqueue(...)
实际插件op的执行函数，我们自己实现的cuda操作就放到这里(当然C++写的op也可以放进来，不过因为是CPU执行，速度就比较慢了)，与往常一样接受输入inputs产生输出outputs，传给相应的指针就可以。

### IPluginV2::getNbOutputs() const  /   IPluginV2::getOutputDimensions
获取层的输出tensor数量/维度。此函数由INetworkDefinition和IBuilder的实现调用。特别是，它在任何对initialize()的调用之前被调用。

### nvinfer1::IPluginV2::getPluginNamespace() const
获取插件op的命名空间

### nvinfer1::IPluginV2::getPluginType() const / nvinfer1::IPluginV2::getPluginVersion() const
获取插件op的类型和版本，用户可以自定义

### nvinfer1::IPluginV2::getSerializationSize()	const
获取插件op的序列化尺寸大小

### nvinfer1::IPluginV2::getWorkspaceSize(int32_t maxBatchSize)	const
这个函数需要返回插件op需要中间显存变量的实际数据大小(bytesize)，可通过TensorRT的接口去获取，是比较规范的方式。
确定插件需要多大的显存空间去运行，在实际运行的时候就可以直接使用TensorRT开辟好的空间而不是自己去申请显存空间。

### nvinfer1::IPluginV2::initialize()
初始化函数，在这个插件准备开始run之前执行。
主要初始化一些提前开辟空间的参数，一般是一些cuda操作需要的参数(例如conv操作需要执行卷积操作，我们就需要提前开辟weight和bias的显存)，假如我们的算子需要这些参数，则在这里需要提前开辟显存。

### nvinfer1::IPluginV2::serialize(...)	const
把需要用的数据按照顺序序列化到buffer里

### nvinfer1::IPluginV2::setPluginNamespace(...)
设置插件op的命名空间

### supportsFormat()
检查数据类型是否正确

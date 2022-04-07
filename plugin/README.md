# TensorRT Plugin接口介绍
插件接口的继承关系

![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/plugin/plugin.jpg)
<br>注意，如果指定了明确的插件batch大小，可以使用IPluginV2，否则使用其他插件接口.
<br>推荐使用IPluginV2DynamicExt，功能完备，可向下兼容。
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

## nvinfer1::IPluginV2Ext()
相比于IPluginV2，该接口支持不同的输出数据类型和跨批次广播。

### nvinfer1::IPluginV2Ext::attachToContext(...)
如果这个op使用到了一些其他东西，例如cublas handle，可以直接借助TensorRT内部提供的cublas handle:

### canBroadcastInputAcrossBatch()
如果插件可以使用跨批次广播而无需复制的输入，则返回 true。

### detachFromContext()
将插件对象从其执行上下文中分离出来。

## nvinfer1::IPluginV2IOEXT()
对比IPluginV2Ext，IPluginV2IOEXT支持更多的tensor类型和I/O类型

### configurePlugin()
配置这个插件op，判断输入和输出类型数量是否正确。官方还提到通过这个配置信息可以告知TensorRT去选择合适的算法(algorithm)去调优这个模型。

### supportsFormatCombination()
TensorRT调用此方法以判断pos索引的输入/输出是否支持inOut[pos].format和inOut[pos].type指定的格式/数据类型。
如果插件支持inOut[pos]处的格式/数据类型，则返回true。 如果是否支持取决于其他的输入/输出格式/数据类型，则插件可以使其结果取决于inOut[0..pos-1]中的格式/数据类型，该格式/数据类型将设置为插件支持的值。 这个函数不需要检查inOut[pos + 1..nbInputs + nbOutputs-1]，pos的决定必须仅基于inOut[0..pos]。

## nvinfer1::IPluginV2DynamicExt()
类似于IPluginV2Ext，但支持动态尺寸，所提供接口与IPluginV2Ext类似，但是传入参数类型不同，详情参考官方文档

### DimsExprs nvinfer1::IPluginV2DynamicExt::getOutputDimensions(...)
获取输出维度

##参考博客/文档
<br>[https://zhuanlan.zhihu.com/p/297002406]
<br>[https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_i_o_ext.html]
<br>[https://blog.csdn.net/xuanwu_yan/article/details/111463822]
<br>[https://zhuanlan.zhihu.com/p/296861242]

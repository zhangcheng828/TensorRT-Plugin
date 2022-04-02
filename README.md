# TensorRT自定义插件指南
TensorRT的源代码仓库结构，下面是我们需要修改的文件

|--TensorRT-main
<br>　|--parsers
<br>　　|--onnx
<br>　　　|--builtin_op_importers.cpp
<br>　　　|--main.cpp
<br>　|--plugin
<br>　　|--CustomPlugin
<br>　　　|--CMakeLists.txt
<br>　　　|--lCustomPlugin.cpp
<br>　　　|--lCustomPlugin.h
<br>　　|--CMakeLists.txt
<br>　　|--InferPlugin.cpp
<br>　|--CMakeLists.txt

## 一、安装TensorRT库
此处略过。自定义插件需要提前安装TensorRT库，还需要下载TensorRT源文件，后续编译要用。

## 二、下载源文件
Github下载TensorRT的编译源文件[链接](https://github.com/NVIDIA/TensorRT)，另外下载第三方库onnx，cub，protobuf并放到TensorRT源文件相应的文件夹里，如下所示：
![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/1.png)
## 三、验证TensorRT是否能正确编译
在TensorRT源文件根目录下执行下列命令：
```bash
cmake -B build
cd build
make
```
如果已安装的tensorRT库未添加环境变量，上述编译过程会报错，提示找不到文件，我们可以直接在CMakeLists.txt文件中指定TensorRT库路径，如下图所示
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/2.png)

另外，官方提供的cmakelists默认编译parser，plugin，还有sample。自定义算子不需要编译sample，可以将它关闭了。
```bash
option(BUILD_PLUGINS "Build TensorRT plugin" ON)
option(BUILD_PARSERS "Build TensorRT parsers" ON)
option(BUILD_SAMPLES "Build TensorRT samples" OFF)
```
最后还有个坑，编译快结束时提示某文件找不到
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/3.png)

在main.cpp里把这行还有对optimize.h调用的代码全都注释掉（对最后的编译结果无影响）。
所有bug解决后就能正常编译了。
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/4.png)

导出结果如下图：
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/5.png)
out文件夹存储了编译后的各种链接库，将其复制到tensorRT/lib目录下，替换原本的链接库文件。
进行到这一步已经可以了，下面开始自定义plugin.

## 四、自定义插件准备工作
准备onnx文件：此处选用PointNet点云分类分割模型，用下面的代码将预训练的pth文件导出onnx
```python
model = readPointNet()
model.eval() 
x=torch.randn((1,3,1024))
torch.onnx.export(model, # 搭建的网络
    x, # 输入张量
    'Pointnet.onnx', # 输出模型名称
    input_names=["input"], # 输入命名
    output_names=["output"], # 输出命名
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
)
```
TensorRT支持PointNet模型中的所有操作，可以直接使用trtexec工具导出。为了增加TRT不支持的op类型，使用下面的脚本将LeakyRelu的op_type改为Custom
```python
import onnx
onnx_model = onnx.load("Pointnet.onnx")
graph = onnx_model.graph
nodes = graph.node
for i in range(len(nodes)):
    if(nodes[i].op_type == "LeakyRelu"):
        nodes[i].op_type = "Custom"
onnx.save(onnx_model,"custom.onnx")
```
<br>这是修改前
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/6.png)
<br>这是修改后

<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/7.png)

最后尝试使用trtexec将修改后的onnx模型导出
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/8.png)

果然报错了
## 五、自定义plugin
打开TRT源代码仓库，复制LeakyReluPlugin文件将其改成CustomPlugin，里面的头文件还有cpp文件统统改为自己想要的名字，顺便把代码里面的类型名也改了，见下图。
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/9.png)

为了后续方便注册，自定义插件Custom类需要继承nvinfer1::IPluginV2DynamicExt接口，另外需要增加一些重写方法和属性，如下：
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/10.png)

下面是对应的cpp实现
```C++
void Custom::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}
const char* Custom::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}
nvinfer1::DataType Custom::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    ASSERT(inputTypes && nbInputs > 0 && index == 0);
    return inputTypes[0];
}
// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void Custom::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept
{
}
// Detach the plugin object from its execution context.
void Custom::detachFromContext() noexcept {}

void Custom::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    // Not support dynamic shape in C dimension
    ASSERT(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
}
```

其他需要实现的方法参考IReluPlugin

另外还需要写一个创建Custom插件的类
```C++
class CustomPluginCreator : public BaseCreator
{
public:
    CustomPluginCreator();

    ~CustomPluginCreator() override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
```
方法实现参考IReluPlugin

## 六、注册plugin
首先在inferplugin.cpp文件中添加初始化插件的接口，如下
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/12.png)

其次在cmake文件中添加编译Custom插件的选项
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/13.png)

最后需要实现onnx结点和TRT插件的映射关系，改动如下：
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/14.png)

## 七、编译运行
回到TRT源代码主目录，使用如下命令编译
```bash
cmake -B build
cd build
make
```
提示如下即代表编译成功
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/15.png)

然后将out文件夹下的库复制到TRT安装目录lib文件夹里。
最后使用官方工具trtexec进行编译，编译成功。
<br>![](https://github.com/zhangcheng828/TensorRT-Plugin/blob/main/figs/16.png)



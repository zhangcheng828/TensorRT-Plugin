# TensorRT自定义插件直接注册不编译TRT源文件踩坑
<br> 使用REGISTER_TENSORRT_PLUGIN进行动态注册
```c++
REGISTER_TENSORRT_PLUGIN(CustomPluginCreator);
```

并将其打包成动态链接库
```bash
cuda_add_library(custom_plugin SHARED
    include/Custom.cu
    include/lCustomPlugin.cpp
  )
```

报错，提示lCustomPlugin.cpp195行传入参数PluginFieldCollection *fc是空值



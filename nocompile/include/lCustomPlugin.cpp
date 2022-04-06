/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <lCustomPlugin.h>
#include <Custom.h>

using namespace nvinfer1;

namespace Customlayer
{
    static const char *Custom_PLUGIN_VERSION{"1"};
    static const char *Custom_PLUGIN_NAME{"Custom"};
    PluginFieldCollection CustomPluginCreator::mFC{};
    std::vector<PluginField> CustomPluginCreator::mPluginAttributes;

    // LeakyReLU {{{
    CustomPlugin::CustomPlugin(float negSlope)
        : mNegSlope(negSlope), mBatchDim(1)
    {
    }

    CustomPlugin::CustomPlugin(const void *buffer, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(buffer), *a = d;
        mNegSlope = read<float>(d);
        mBatchDim = read<int>(d);
        assert(d == a + length);
    }

    int CustomPlugin::getNbOutputs() const noexcept
    {
        return 1;
    }

    // Dims CustomPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
    // {
    //     ASSERT(nbInputDims == 1);
    //     ASSERT(index == 0);
    //     return inputs[0];
    // }

    nvinfer1::DimsExprs CustomPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                          int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept
    {
        nvinfer1::DimsExprs output(inputs[0]);
        return output;
    }

    int CustomPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                              const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace,
                              cudaStream_t stream) noexcept
    {
        const void *inputData = inputs[0];
        void *outputData = outputs[0];
        int status = CustomInference(stream, mBatchDim, mNegSlope, inputData, outputData);
        return status;
    }

    size_t CustomPlugin::getSerializationSize() const noexcept
    {
        // mNegSlope, mBatchDim
        return sizeof(float) + sizeof(int);
    }

    void CustomPlugin::serialize(void *buffer) const noexcept
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, mNegSlope);
        write(d, mBatchDim);
        assert(d == a + getSerializationSize());
    }

    bool CustomPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut,
                                                 int nbInputs, int nbOutputs) noexcept
    {
        assert(0 <= pos && pos < 2);
        const auto *in = inOut;
        const auto *out = inOut + nbInputs;
        switch (pos)
        {
        case 0:
            return in[0].type == DataType::kFLOAT &&
                   in[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 1:
            return out[0].type == in[0].type &&
                   out[0].format == nvinfer1::TensorFormat::kLINEAR;
        }
    }

    int CustomPlugin::initialize() noexcept
    {
        return 0;
    }

    void CustomPlugin::terminate() noexcept {}

    size_t CustomPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                          int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept
    {
        return 0;
    }

    const char *CustomPlugin::getPluginType() const noexcept
    {
        return Custom_PLUGIN_NAME;
    }

    const char *CustomPlugin::getPluginVersion() const noexcept
    {
        return Custom_PLUGIN_VERSION;
    }

    void CustomPlugin::destroy() noexcept
    {
        delete this;
    }

    nvinfer1::IPluginV2DynamicExt *CustomPlugin::clone() const noexcept
    {
        auto *plugin = new CustomPlugin(mNegSlope);
        plugin->setPluginNamespace(mPluginNamespace.c_str());
        plugin->initialize();
        return plugin;
    }

    void CustomPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
    {
        mPluginNamespace = pluginNamespace;
    }

    const char *CustomPlugin::getPluginNamespace() const noexcept
    {
        return mPluginNamespace.c_str();
    }

    nvinfer1::DataType CustomPlugin::getOutputDataType(
        int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept
    {
        assert(inputTypes && nbInputs > 0 && index == 0);
        return inputTypes[0];
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void CustomPlugin::attachToContext(
        cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) noexcept
    {
    }

    // Detach the plugin object from its execution context.
    void CustomPlugin::detachFromContext() noexcept {}

    void CustomPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept
    {
        // Not support dynamic shape in C dimension
        assert(nbInputs == 1 && in[0].desc.dims.d[1] != -1);
    }

    CustomPluginCreator::CustomPluginCreator()
    {
        mPluginAttributes.clear();
        mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char *CustomPluginCreator::getPluginName() const noexcept
    {
        return Custom_PLUGIN_NAME;
    }

    const char *CustomPluginCreator::getPluginVersion() const noexcept
    {
        return Custom_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection *CustomPluginCreator::getFieldNames() noexcept
    {
        return &mFC;
    }

    nvinfer1::IPluginV2DynamicExt *CustomPluginCreator::createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept
    {
        const nvinfer1::PluginField *fields = fc->fields;
        std::cout << fc->nbFields << std::endl;
        assert(fc->nbFields == 1);
        assert(fields[0].type == PluginFieldType::kFLOAT32);
        float negSlope = *(static_cast<const float *>(fields[0].data));
        CustomPlugin *obj = new CustomPlugin{negSlope};

        obj->setPluginNamespace(mNamespace.c_str());
        obj->initialize();
        return obj;

        // (void)name;
        // (void)fc;
        // return nullptr;
    }

    nvinfer1::IPluginV2DynamicExt *CustomPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
    {
        // This object will be deleted when the network is destroyed, which will
        // call CustomPlugin::destroy()
        CustomPlugin *obj = new CustomPlugin{serialData, serialLength};
        obj->setPluginNamespace(mNamespace.c_str());
        obj->initialize();
        return obj;
    }
}

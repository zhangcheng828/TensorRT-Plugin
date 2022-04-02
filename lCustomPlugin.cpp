#include "lCustomPlugin.h"
#include "checkMacrosPlugin.h"
#include "kernel.h"

using namespace nvinfer1;
using nvinfer1::plugin::CustomPluginCreator;
using nvinfer1::plugin::Custom;

static const char* Custom_PLUGIN_VERSION{"1"};
static const char* Custom_PLUGIN_NAME{"Custom_TRT"};
PluginFieldCollection CustomPluginCreator::mFC{};
std::vector<PluginField> CustomPluginCreator::mPluginAttributes;

// LeakyReLU {{{
Custom::Custom(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
}

Custom::Custom(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int Custom::getNbOutputs() const noexcept
{
    return 1;
}

DimsExprs Custom::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
                int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output(inputs[0]);
    return output;
}

int Custom::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    //在common/kernel.h文件里按照IReluInference实现CustomInference
    pluginStatus_t status = CustomInference(stream, mBatchDim, mNegSlope, inputData, outputData);
    return status;
}

size_t Custom::getSerializationSize() const noexcept
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void Custom::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

bool Custom::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, 
        int nbInputs, int nbOutputs) noexcept
{
    assert(0 <= pos && pos < 2);
    const auto *in = inOut;
    const auto *out = inOut + nbInputs;
    switch (pos) {
        case 0:
        return in[0].type == DataType::kFLOAT &&
                in[0].format == nvinfer1::TensorFormat::kLINEAR;
        case 1:
        return out[0].type == in[0].type &&
                out[0].format == nvinfer1::TensorFormat::kLINEAR;
    }
}


int Custom::initialize() noexcept
{
    return 0;
}

void Custom::terminate() noexcept {}

size_t Custom::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
            int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

const char* Custom::getPluginType() const noexcept
{
    return Custom_PLUGIN_NAME;
}

const char* Custom::getPluginVersion() const noexcept
{
    return Custom_PLUGIN_VERSION;
}

void Custom::destroy() noexcept
{
    delete this;
}

IPluginV2DynamicExt* Custom::clone() const noexcept
{
    auto* plugin = new Custom(mNegSlope);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    plugin->initialize();
    return plugin;
}

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

CustomPluginCreator::CustomPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("negSlope", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CustomPluginCreator::getPluginName() const noexcept
{
    return Custom_PLUGIN_NAME;
}

const char* CustomPluginCreator::getPluginVersion() const noexcept
{
    return Custom_PLUGIN_VERSION;
}

const PluginFieldCollection* CustomPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* CustomPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kFLOAT32);
    float negSlope = *(static_cast<const float*>(fields[0].data));
    Custom* obj = new Custom{negSlope};

    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

IPluginV2DynamicExt* CustomPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call CustomPlugin::destroy()
    Custom* obj = new Custom{serialData, serialLength};
    obj->setPluginNamespace(mNamespace.c_str());
    obj->initialize();
    return obj;
}

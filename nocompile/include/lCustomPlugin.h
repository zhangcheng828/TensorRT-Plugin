
#ifndef TRT_L_Custom_PLUGIN_H
#define TRT_L_Custom_PLUGIN_H
// #include "NvInfer.h"
#include <NvInferPlugin.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>

namespace Customlayer
{

    template <typename T>
    T read(const char *&buffer)
    {
        T val{};
        std::memcpy(&val, buffer, sizeof(T));
        buffer += sizeof(T);
        return val;
    }

    template <typename T>
    void write(char *&buffer, const T &val)
    {
        std::memcpy(buffer, &val, sizeof(T));
        buffer += sizeof(T);
    }

    class CustomPlugin : public nvinfer1::IPluginV2DynamicExt
    {
    public:
        CustomPlugin(float negSlope);

        CustomPlugin(const void *buffer, size_t length);

        CustomPlugin() = delete;

        ~CustomPlugin() override = default;

        int getNbOutputs() const noexcept override;

        nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept override;

        int initialize() noexcept override;

        void terminate() noexcept override;

        size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                int nbInputs, const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const noexcept override;

        int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;

        size_t getSerializationSize() const noexcept override;

        void serialize(void *buffer) const noexcept override;

        bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *inOut,
                                       int nbInputs, int nbOutputs) noexcept override;
        const char *getPluginType() const noexcept override;

        const char *getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;

        void setPluginNamespace(const char *pluginNamespace) noexcept override;

        const char *getPluginNamespace() const noexcept override;

        nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const noexcept override;

        void attachToContext(cudnnContext *cudnn, cublasContext *cublas, nvinfer1::IGpuAllocator *allocator) noexcept override;
        void detachFromContext() noexcept override;

        void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in, int nbInputs,
                             const nvinfer1::DynamicPluginTensorDesc *out, int nbOutputs) noexcept override;

    private:
        float mNegSlope;
        int mBatchDim;
        std::string mPluginNamespace;
        std::string mNamespace;
    };

    class CustomPluginCreator : public nvinfer1::IPluginCreator
    {
    public:
        CustomPluginCreator();

        ~CustomPluginCreator() override = default;

        const char *getPluginName() const noexcept override;

        const char *getPluginVersion() const noexcept override;

        const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

        nvinfer1::IPluginV2DynamicExt *createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept override;

        nvinfer1::IPluginV2DynamicExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char *libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        const char *getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    private:
        std::string mNamespace;
        static nvinfer1::PluginFieldCollection mFC;
        static std::vector<nvinfer1::PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(CustomPluginCreator);
}

#endif // TRT_L_RELU_PLUGIN_H

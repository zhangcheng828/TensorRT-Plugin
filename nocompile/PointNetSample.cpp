#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include "TRTModelBase.h"
#include "lCustomPlugin.h"


class TRTModelPointNet : public TRTModelBase
{
private:
    int outputClsSize;              //!< The number of output classes

protected:
    bool processInput(const trtCommon::BufferManager& buffers) override;
    bool processOutput(const trtCommon::BufferManager& buffers) override;

public:
    TRTModelPointNet(const ConfigParams params): TRTModelBase(params)
    {}

};


bool TRTModelPointNet::processInput(const trtCommon::BufferManager& buffers)
{
    return true;
}
bool TRTModelPointNet::processOutput(const trtCommon::BufferManager& buffers)
{
    return true;
}

ConfigParams initializeParams()
{
    ConfigParams params;

    params.data_dirs.push_back("data/PointNet/");
    
    params.model_type = ModelType::Onnx;
    // params.prototxtFileName = "ssd.prototxt";
    params.file_path = "custom.onnx";
    params.inputTensorNames.push_back("actual_input_1");
    params.batch_size = 1;
    params.engine_path = "../serialize_engine/serialize_engine_output.trt";
    params.load_engine = false;
    params.dla_core = -1;
    params.MaxWorkspaceSize = 960_MiB;

    return params;
}

int main()
{
    auto params = initializeParams();
    TRTModelPointNet SSD{params};

    std::cout << "Building and running a GPU inference engine for SSD" << std::endl;


    // if (!SSD.forward())
    // {
    //     // return print::gLogger.reportFail(sampleTest);
    // }
    // if(!params.load_engine && !SSD.saveEngine())
    // {
    //     // return print::gLogger.reportFail(sampleTest);
    // }
    return 0;
}

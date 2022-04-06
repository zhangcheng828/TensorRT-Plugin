#ifndef CUSTOM_HPP_
#define CUSTOM_HPP_
#include <cuda_runtime_api.h>
namespace Customlayer
{
int CustomInference(cudaStream_t stream, int n, float negativeSlope, const void* input, void* output);
}
#endif  // MISH_HPP_

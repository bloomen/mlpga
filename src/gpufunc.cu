#include <stdexcept>

#include <cuda.h>
#include <curand_kernel.h>

namespace mlpga
{

namespace gpu
{

namespace kernel
{

__global__ void sum(float* result, const float* data, const std::size_t n) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        atomicAdd(result, data[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void divide_by(float* data, const std::size_t n, const float value) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        data[tid] /= value;
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void setup_rand(curandState* state)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, tid, 0, &state[tid]);
}

__global__ void give_birth(float* w1, float* w2, std::size_t n,
                           float crossover_ratio, float mutate_ratio,
                           float mutate_sigma, curandState* d_state)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        if (curand_uniform(d_state + tid) < crossover_ratio)
        {
            auto tmp = w1[tid];
            w1[tid] = w2[tid];
            w2[tid] = tmp;
        }
        if (curand_uniform(d_state + tid) < mutate_ratio)
        {
            w1[tid] += w1[tid] * (curand_uniform(d_state + tid) - 0.5f);
        }
        if (curand_uniform(d_state + tid) < mutate_ratio)
        {
            w2[tid] += w2[tid] * (curand_uniform(d_state + tid) - 0.5f);
        }
        tid += blockDim.x * gridDim.x;
    }
}

}

void sum(cudaStream_t& stream, float& result, const float& data, std::size_t n) {
    if (n == 0) {
        throw std::invalid_argument("n must be larger than zero");
    }
    kernel::sum<<<128, 128, 0, stream>>>(&result, &data, n);
}

void divide_by(cudaStream_t& stream, float& data, const std::size_t n, const float value) {
    if (n == 0) {
        throw std::invalid_argument("n must be larger than zero");
    }
    if (value == 0) {
        throw std::invalid_argument("value cannot be zero");
    }
    kernel::divide_by<<<128, 128, 0, stream>>>(&data, n, value);
}

#define MIN 2
#define MAX 7
#define ITER 10000000

void give_birth(cudaStream_t& stream, float* w1, float* w2, std::size_t n,
                float crossover_ratio, float mutate_ratio, float mutate_sigma)
{
    curandState* d_state;
    cudaMalloc(&d_state, sizeof(curandState));
    kernel::setup_rand<<<128, 128, 0, stream>>>(d_state);
    kernel::give_birth<<<128, 128, 0, stream>>>(w1, w2, n, crossover_ratio, mutate_ratio, mutate_sigma, d_state);
    cudaFree(d_state);
}

}

}

#include "gpufunc.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cuda.h>
#include <curand_kernel.h>

#define CUASSERT(ans) { cuda_assert((ans), __FILE__, __LINE__); }

namespace
{

inline
void cuda_assert(const cudaError_t code, const char* const file,
                 const int line, const bool abort=true)
{
   if (code != cudaSuccess)
   {
      std::ostringstream os;
      os << "cuda_assert: " << cudaGetErrorString(code) << " " << file << ":" << line ;
      std::cerr << os.str() << std::endl;
      if (abort)
      {
          assert(false);
          throw std::runtime_error(os.str());
      }
   }
}

cudaStream_t from(const mlpga::gpu::Stream& stream)
{
    return reinterpret_cast<cudaStream_t>(stream.get());
}

curandGenerator_t from(const mlpga::gpu::RandomState& random_state)
{
    return reinterpret_cast<curandGenerator_t>(random_state.get());
}

}

namespace mlpga
{

namespace gpu
{

namespace kernel
{

__global__ void crossover(float* const w1, float* const w2, const std::size_t n,
                          const float crossover_ratio, const float* const rnd)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n)
    {
        if (rnd[tid] < crossover_ratio)
        {
            const auto tmp = w1[tid];
            w1[tid] = w2[tid];
            w2[tid] = tmp;
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void mutate(float* const w, const std::size_t n,
                       const float mutate_ratio,
                       const float mutate_scale,
                       const float* const rnd_ratio,
                       const float* const rnd_scale)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n)
    {
        if (rnd_ratio[tid] < mutate_ratio)
        {
            w[tid] += w[tid] * (rnd_scale[tid] - 0.5f) * mutate_scale;
        }
        tid += blockDim.x * gridDim.x;
    }
}

}

struct Stream::impl
{
    impl()
    {
        CUASSERT(cudaStreamCreate(&stream));
    }
    ~impl()
    {
        CUASSERT(cudaStreamDestroy(stream));
    }
    cudaStream_t stream;
};

Stream::Stream()
    : impl_{new impl}
{}

Stream::~Stream()
{}

void* Stream::get() const
{
    return reinterpret_cast<void*>(impl_->stream);
}

void Stream::sync()
{
    CUASSERT(cudaStreamSynchronize(impl_->stream));
}

Array::Array(Stream& s, const std::size_t size, const bool async)
    : stream_{&s}
    , size_{size}
{
    if (async)
    {
        CUASSERT(cudaMallocHost(&host_, size_ * sizeof(float)));
    }
    CUASSERT(cudaMalloc(&device_, size_ * sizeof(float)));
}

Array::~Array()
{
    CUASSERT(cudaFree(device_));
    if (host_)
    {
        CUASSERT(cudaFreeHost(host_));
    }
}

const float* Array::device() const
{
    return device_;
}

float* Array::device()
{
    return device_;
}

const float* Array::host() const
{
    return host_;
}

float* Array::host()
{
    return host_;
}

std::size_t Array::size() const
{
    return size_;
}

void Array::copy_to_device()
{
    assert(host_);
    CUASSERT(cudaMemcpyAsync(device_, host_, sizeof(float) * size_, cudaMemcpyHostToDevice, from(*stream_)));
}

void Array::copy_to_host()
{
    assert(host_);
    CUASSERT(cudaMemcpyAsync(host_, device_, sizeof(float) * size_, cudaMemcpyDeviceToHost, from(*stream_)));
}

struct RandomState::impl
{
    impl()
    {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    }
    ~impl()
    {
        curandDestroyGenerator(gen);
    }
    curandGenerator_t gen;
};

RandomState::RandomState(Stream& stream, const std::size_t seed)
    : impl_{new impl}
{
    curandSetPseudoRandomGeneratorSeed(impl_->gen, seed);
    curandSetStream(impl_->gen, from(stream));
}

RandomState::~RandomState()
{}

void RandomState::generate(Array& array)
{
    curandGenerateUniform(impl_->gen, array.device(), array.size());
}

void* RandomState::get() const
{
    return reinterpret_cast<void*>(impl_->gen);
}

void device_sync()
{
    CUASSERT(cudaDeviceSynchronize());
}

void crossover(Stream& stream, Array& w1, Array& w2,
               const float crossover_ratio, const Array& rnd)
{
    const auto n_blocks = (w1.size() + 127) / 128;
    kernel::crossover<<<n_blocks, 128, 0, from(stream)>>>(w1.device(), w2.device(), w1.size(),
                                                          crossover_ratio, rnd.device());
}

void mutate(Stream& stream, Array& w,
            const float mutate_ratio, const float mutate_scale,
            const Array& rnd_ratio, const Array& rnd_scale)
{
    const auto n_blocks = (w.size() + 127) / 128;
    kernel::mutate<<<n_blocks, 128, 0, from(stream)>>>(w.device(), w.size(),
                                                       mutate_ratio, mutate_scale,
                                                       rnd_ratio.device(), rnd_scale.device());
}

}

}

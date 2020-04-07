#pragma once

#include <cstddef>

class cudaStream_t;

namespace mlpga
{

namespace gpu
{

void sum(cudaStream_t& stream, float& result, const float& data, std::size_t n);

void divide_by(cudaStream_t& stream, float& data, std::size_t n, float value);

void give_birth(cudaStream_t& stream, float* w1, float* w2, std::size_t n,
                float crossover_ratio, float mutate_ratio, float mutate_sigma);

}

}

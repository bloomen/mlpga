#pragma once

#include <ctime>
#include <random>

namespace mlpga
{

inline long unsigned int time_seed()
{
    return static_cast<long unsigned int>(std::time(nullptr));
}

namespace detail
{

template<typename RandomEngine>
inline void xavier(float* weights,
                   const std::size_t n,
                   RandomEngine& random_engine)
{
    std::normal_distribution<float> normal{0.0f, 1.0f / static_cast<float>(n - 1)}; // bias not included
    for (std::size_t i = 0; i < n; ++i)
    {
        weights[i] = normal(random_engine);
    }
}

}

}

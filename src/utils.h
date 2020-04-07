#pragma once

#include <vector>

#include "init.h"

namespace mlpga
{

struct Split
{
    std::vector<std::vector<float>> X_train;
    std::vector<std::vector<float>> X_test;
    std::vector<std::vector<float>> y_train;
    std::vector<std::vector<float>> y_test;
};

template<typename RandomEngine>
inline Split split_train_test(const std::vector<std::vector<float>>& X,
                              const std::vector<std::vector<float>>& y,
                              const float test_ratio,
                              RandomEngine& random_engine)
{
    std::uniform_real_distribution<float> uniform;
    Split split;
    for (std::size_t i = 0; i < X.size(); ++i)
    {
        if (uniform(random_engine) < test_ratio)
        {
            split.X_test.push_back(X[i]);
            split.y_test.push_back(y[i]);
        }
        else
        {
            split.X_train.push_back(X[i]);
            split.y_train.push_back(y[i]);
        }
    }
    return split;
}

inline float mae(const float* const truth,
                 const float* const pred,
                 const std::size_t n)
{
    assert(truth != nullptr);
    assert(pred != nullptr);
    assert(n > 0);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
    {
        sum += std::abs(truth[i] - pred[i]);
    }
    return sum / static_cast<float>(n);
}

inline std::vector<float> flatten(const std::vector<std::vector<float>>& data)
{
    std::vector<float> res;
    res.reserve(data.size() * data.front().size());
    for (const auto row : data)
    {
        for (const auto value : row)
        {
            res.push_back(value);
        }
    }
    return res;
}

}

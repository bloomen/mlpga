#pragma once

#include <random>

namespace mlpga
{

class RandomEngine
{
public:
    using result_type = unsigned long;
    virtual ~RandomEngine() = default;
    virtual result_type operator()() = 0;
    virtual result_type min() const = 0;
    virtual result_type max() const = 0;
};

class DefaultRandomEngine : public RandomEngine
{
public:
    explicit
    DefaultRandomEngine(const std::size_t seed)
        : random_engine_{seed}
    {}
    result_type operator()() override
    {
        return random_engine_();
    }
    result_type min() const override
    {
        return std::default_random_engine::min();
    }
    result_type max() const override
    {
        return std::default_random_engine::max();
    }
private:
    std::default_random_engine random_engine_;
};

namespace detail
{

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

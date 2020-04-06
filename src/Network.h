#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include "init.h"
#include "utils.h"

namespace mlpga
{

struct Target
{
    enum Type : std::uint8_t
    {
        Classification,
        Regression,
    };

    Target(const Type type,
           const float class0 = 0.0f,
           const float class1 = 1.0f)
        : type{type}
        , class0{class0}
        , class1{class1}
    {}

    Type type;
    float class0;
    float class1;
};

using Printer = std::function<void(const std::string& value)>;
using Writer = std::function<void(const char* value, std::size_t size)>;
using Reader = std::function<void(char* value, std::size_t size)>;

namespace detail
{

inline void softmax(float* output, const std::size_t n)
{
    const auto D = *std::max_element(output, output + n);
    float denom = 1e-6f;
    for (std::size_t i = 0; i < n; ++i)
    {
        output[i] -= D; // for numerical stability
        denom += std::exp(output[i]);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        output[i] = std::exp(output[i]) / denom;
    }
}

inline void sigmoid(float& x)
{
    x = 1.0f / (1.0f + std::exp(-x));
}

inline void relu(float& x)
{
    if (x < 0.0f)
    {
        x = 0.0f;
    }
}

inline float activate(const float* const weights,
                      const std::vector<float>& inputs)
{
    float output = 0.0f;
    for (std::size_t i = 0; i < inputs.size(); ++i)
    {
        output += weights[i] * inputs[i];
    }
    output += weights[inputs.size()]; // bias
    return output;
}

template<typename T>
void write(Writer& writer, const T& value)
{
    writer(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
void read(Reader& reader, T& value)
{
    reader(reinterpret_cast<char*>(&value), sizeof(T));
}

}

class Network
{
public:
    Network(Target target = Target{Target::Regression},
            std::vector<std::size_t> layers = {1},
            RandomEngine* random_engine = nullptr)
        : target_{std::move(target)}
        , layers_{std::move(layers)}
    {
        assert(layers_.size() > 0);
        assert(target_.class0 < target_.class1);

        std::size_t weight_count = 0;
        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            if (i == 0)
            {
                weight_count += 2 * layers_[i];
            }
            else
            {
                weight_count += layers_[i - 1] * layers_[i] + layers_[i];
            }
        }
        weights_.resize(weight_count);

        if (random_engine)
        {
            init_weights(*random_engine);
        }
    }

    void init_weights(RandomEngine& random_engine)
    {
        auto weights = weights_.data();
        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            for (std::size_t j = 0; j < layers_[i]; ++j)
            {
                std::size_t weight_count;
                if (i == 0)
                {
                    weight_count = 2;
                }
                else
                {
                    weight_count = layers_[i - 1] + 1;
                }
                detail::xavier(weights, weight_count, random_engine);
                weights += weight_count;
            }
        }
        assert(weights == weights_.data() + weights_.size());
    }

    std::string arch_string() const
    {
        std::string arch = "Network arch (" + std::to_string(layers_.size()) + "): ";
        for (const auto& layer : layers_)
        {
            arch += std::to_string(layer);
            if (&layer != &layers_.back())
            {
                arch += " -> ";
            }
        }
        return arch;
    }

    const Target& get_target() const
    {
        return target_;
    }

    const std::vector<std::size_t>& get_layers() const
    {
        return layers_;
    }

    const std::vector<float>& get_weights() const
    {
        return weights_;
    }

    std::vector<float>& get_weights()
    {
        return weights_;
    }

    void set_weights(std::vector<float> weights)
    {
        assert(weights_.size() == weights.size());
        weights_ = std::move(weights);
    }

    void save(Writer& writer) const
    {
        detail::write(writer, target_.type);
        detail::write(writer, target_.class0);
        detail::write(writer, target_.class1);
        detail::write(writer, layers_.size());
        for (const auto& layer : layers_)
        {
            detail::write(writer, layer);
        }
        detail::write(writer, weights_.size());
        for (const auto& value : weights_)
        {
            detail::write(writer, value);
        }
    }

    static Network load(Reader& reader)
    {
        Target target{Target::Regression};
        detail::read(reader, target.type);
        detail::read(reader, target.class0);
        detail::read(reader, target.class1);
        std::size_t layer_count;
        detail::read(reader, layer_count);
        std::vector<std::size_t> layers(layer_count);
        for (auto& value : layers)
        {
            detail::read(reader, value);
        }
        std::size_t weight_count;
        detail::read(reader, weight_count);
        std::vector<float> weights(weight_count);
        for (auto& value : weights)
        {
            detail::read(reader, value);
        }
        Network net{std::move(target), std::move(layers)};
        net.set_weights(std::move(weights));
        return net;
    }

    std::vector<float> predict(const std::vector<float>& input) const
    {
        std::vector<float> output = input;
        std::vector<float> new_input;
        auto weights = weights_.data();

        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            new_input.reserve(layers_[i]);
            for (std::size_t j = 0; j < layers_[i]; ++j)
            {
                auto value = detail::activate(weights, output);
                if (i < layers_.size() - 1)
                {
                    detail::relu(value);
                }
                new_input.push_back(value);
                if (i == 0)
                {
                    weights += 2;
                }
                else
                {
                    weights += layers_[i - 1] + 1;
                }
            }
            output = std::move(new_input);
        }
        assert(weights == weights_.data() + weights_.size());

        if (target_.type == Target::Classification)
        {
            if (output.size() == 1)
            {
                detail::sigmoid(output.front());
            }
            else
            {
                detail::softmax(output.data(), output.size());
            }
            for (auto& value : output)
            {
                value = (target_.class1 - target_.class0) * value + target_.class0;
            }
        }
        return output;
    }

private:
    Target target_;
    std::vector<std::size_t> layers_;
    std::vector<float> weights_;
};

}

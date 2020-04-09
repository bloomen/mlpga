#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <type_traits>

#include "init.h"
#include "math.h"
#include "utils.h"

namespace mlpga
{

using Printer = std::function<void(const std::string& value)>;
using Writer = std::function<void(const char* value, std::size_t size)>;
using Reader = std::function<void(char* value, std::size_t size)>;

namespace detail
{

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
    template<typename RandomEngine = std::default_random_engine>
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

    template<typename RandomEngine>
    void init_weights(RandomEngine& random_engine)
    {
        auto weights = weights_.data();
        for (std::size_t i = 0; i < layers_.size(); ++i)
        {
            const auto weight_count = i == 0 ? 2u : layers_[i - 1] + 1;
            for (std::size_t j = 0; j < layers_[i]; ++j)
            {
                detail::xavier(weights, weight_count, random_engine);
                weights += weight_count;
            }
        }
        assert(weights == weights_.data() + weights_.size());
    }

    std::string arch_string() const
    {
        std::string arch = "Network arch (" + std::to_string(weights_.size()) + "): ";
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

    void predict(float* const output, const float* const input) const
    {
        static thread_local std::vector<float> new_in(*std::max_element(layers_.begin(), layers_.end()));
        static thread_local std::vector<float> new_out(new_in.size());
        detail::predict(output, input, target_, weights_.data(), weights_.size(), layers_.data(), layers_.size(), new_in.data(), new_out.data());
    }

private:
    Target target_;
    std::vector<std::size_t> layers_;
    std::vector<float> weights_;
};

}

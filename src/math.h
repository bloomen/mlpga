#pragma once

namespace mlpga
{

struct Target
{
    enum Type : uint8_t
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

namespace detail
{

inline
void min_max(const float* const data,
             const size_t n,
             float& minimum,
             float& maximum)
{
    minimum = 1e9f;
    maximum = -1e9f;
    for (size_t i = 0; i < n; ++i)
    {
        if (data[i] < minimum)
        {
            minimum = data[i];
        }
        if (data[i] > maximum)
        {
            maximum = data[i];
        }
    }
}

inline
void softmax(float* const output, const size_t n)
{
    float minimum;
    float D;
    min_max(output, n, minimum, D);
    float denom = 1e-6f;
    for (size_t i = 0; i < n; ++i)
    {
        output[i] -= D; // for numerical stability
        denom += exp(output[i]);
    }
    for (size_t i = 0; i < n; ++i)
    {
        output[i] = exp(output[i]) / denom;
    }
}

inline
void sigmoid(float& x)
{
    x = 1.0f / (1.0f + exp(-x));
}

inline
void relu(float& x)
{
    if (x < 0.0f)
    {
        x = 0.0f;
    }
}

inline
float activate(const float* const weights,
               const float* const inputs,
               const size_t n)
{
    float output = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        output += weights[i] * inputs[i];
    }
    output += weights[n]; // bias
    return output;
}

inline
void copy(const float* const input,
          const size_t size,
          float* const output)
{
    for (size_t i = 0; i < size; ++i)
    {
        output[i] = input[i];
    }
}

inline
void predict(float* const output,
             const float* const input,
             const Target& network_target,
             const float* const network_weights,
             const size_t n_weights,
             const size_t* const network_layers,
             const size_t n_layers,
             float* const in_cache,
             float* const out_cache)
{
    assert(n_weights > 0);
    assert(n_layers > 0);
    const auto input_size = network_layers[0];
    const auto output_size = network_layers[n_layers - 1];

    auto weights = network_weights;
    size_t current_size = input_size;
    size_t weight_count = 2u;
    const float* source = input;
    for (size_t i = 0; i < n_layers; ++i)
    {
        auto target = i == n_layers - 1 ? output : out_cache;
        for (size_t j = 0; j < network_layers[i]; ++j)
        {
            auto value = detail::activate(weights, source, current_size);
            if (i < n_layers - 1)
            {
                detail::relu(value);
            }
            target[j] = value;
            weights += weight_count;
        }
        if (i < n_layers - 1)
        {
            current_size = network_layers[i];
            weight_count = current_size + 1;
            source = in_cache;
            detail::copy(target, network_layers[i], in_cache);
        }
    }
    assert(weights == network_weights + n_weights);

    if (network_target.type == Target::Classification)
    {
        if (output_size == 1)
        {
            detail::sigmoid(*output);
        }
        else
        {
            detail::softmax(output, output_size);
        }
        const auto factor = network_target.class1 - network_target.class0;
        const auto offset = network_target.class0;
        for (size_t i = 0; i < output_size; ++i)
        {
            output[i] *= factor;
            output[i] += offset;
        }
    }
}

}

}

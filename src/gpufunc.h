#pragma once

#include <cstddef>
#include <memory>

namespace mlpga
{

namespace gpu
{

class Stream
{
public:
    Stream();
    ~Stream();

    void* get() const;

    void sync();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

class Array
{
public:
    Array(Stream& s, std::size_t size, bool async = false);
    ~Array();

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&&) = default;
    Array& operator=(Array&&) = default;

    const float* device() const;
    float* device();

    const float* host() const;
    float* host();

    std::size_t size() const;

    void copy_to_device();
    void copy_to_host();

private:
    Stream* stream_;
    float* host_ = nullptr;
    float* device_;
    std::size_t size_;
};

class RandomState
{
public:
    RandomState(Stream& stream, std::size_t seed);
    ~RandomState();

    void generate(Array& array);

    void* get() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

void device_sync();

void crossover(Stream& stream, Array& w1, Array& w2,
               const float crossover_ratio, const Array& rnd);

void mutate(Stream& stream, Array& w,
            const float mutate_ratio, const float mutate_scale,
            const Array& rnd_ratio, const Array& rnd_scale);

}

}

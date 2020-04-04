#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "init.h"
#include "Network.h"
#include "utils.h"

namespace mlpga
{

inline void crossover(float* const w1,
                      float* const w2,
                      const std::size_t n,
                      const float ratio,
                      RandomEngine& random_engine)
{
    assert(w1 != nullptr);
    assert(w2 != nullptr);
    assert(n > 0);
    assert(ratio > 0.0f);
    assert(ratio < 1.0f);
    std::uniform_real_distribution<float> uniform;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (uniform(random_engine) < ratio)
        {
            std::swap(w1[i], w2[i]);
        }
    }
}

inline void mutate(float* const w,
                   const std::size_t n,
                   const float ratio,
                   const float sigma,
                   RandomEngine& random_engine)
{
    assert(w != nullptr);
    assert(n > 0);
    assert(ratio > 0.0f);
    assert(ratio < 1.0f);
    assert(sigma > 0.0f);
    std::uniform_real_distribution<float> uniform;
    std::normal_distribution<float> normal(0.0f, sigma);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (uniform(random_engine) < ratio)
        {
            w[i] += w[i] * normal(random_engine);
        }
    }
}

struct Model
{
    explicit
    Model(Network network)
        : network{std::move(network)}
    {}
    std::optional<float> fitness;
    Network network;
};

inline std::vector<Model> make_population(const std::size_t size,
                                          const Network& network)
{
    std::vector<Model> population;
    population.reserve(size);
    for (std::size_t p = 0; p < size; ++p)
    {
        population.emplace_back(network.clone());
    }
    return population;
}

inline void select_fittest(std::vector<Model>& population,
                           const std::size_t n_fittest,
                           const std::function<float(const float*, const float*, std::size_t)> fitness,
                           const std::vector<std::vector<float>>& X,
                           const std::vector<float>& y)
{
    for (auto& model : population)
    {
        if (!model.fitness.has_value())
        {
            std::vector<float> pred;
            for (const auto& row : X)
            {
                for (const auto value : model.network.predict(row))
                {
                    pred.push_back(value);
                }
            }
            assert(y.size() == pred.size());
            model.fitness = fitness(y.data(), pred.data(), y.size());
        }
    }
    std::sort(population.begin(), population.end(), [](const auto& x, const auto& y)
    {
        return *x.fitness < *y.fitness;
    });
    population.erase(population.begin() + n_fittest, population.end());
}

inline void reproduce(std::vector<Model>& population,
                      const float crossover_ratio,
                      const float mutate_ratio,
                      const float mutate_sigma,
                      RandomEngine& random_engine)
{
    const auto size = population.size();
    assert(size % 2 == 0);
    for (std::size_t i = 0; i < size; i += 2)
    {
        auto child1 = population[i].network.clone();
        auto child2 = population[i + 1].network.clone();
        crossover(child1.get_weights().data(),
                  child2.get_weights().data(),
                  child1.get_weights().size(),
                  crossover_ratio,
                  random_engine);
        mutate(child1.get_weights().data(),
               child1.get_weights().size(),
               mutate_ratio,
               mutate_sigma,
               random_engine);
        mutate(child2.get_weights().data(),
               child2.get_weights().size(),
               mutate_ratio,
               mutate_sigma,
               random_engine);
        population.emplace_back(std::move(child1));
        population.emplace_back(std::move(child2));
    }
}

inline std::vector<Model> ga_optimize(const Network& network,
                                      const std::size_t n_generations,
                                      const std::size_t population_size,
                                      const float crossover_ratio,
                                      const float mutate_ratio,
                                      const float mutate_sigma,
                                      const std::vector<std::vector<float>>& X,
                                      const std::vector<std::vector<float>>& y,
                                      Printer& printer,
                                      RandomEngine& random_engine)
{
    const auto n_fittest = population_size / 2;
    auto population = mlpga::make_population(n_fittest, network);
    const auto y_ref = flatten(y);
    for (std::size_t g = 0; g < n_generations; ++g)
    {
        mlpga::reproduce(population, crossover_ratio, mutate_ratio, mutate_sigma, random_engine);
        printer("Generation: " + std::to_string(g) + "\n");
        printer("Population size: " + std::to_string(population.size()) + "\n");
        mlpga::select_fittest(population, n_fittest, mae, X, y_ref);
        printer("Best fitness: " + std::to_string(*population.front().fitness) + "\n");
    }
    return population;
}

}

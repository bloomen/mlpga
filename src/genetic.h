#pragma once

#define TRANSWARP_MINIMUM_TASK_SIZE
#include "transwarp.h"
namespace tw = transwarp;

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
    Model(Network network = {})
        : network{std::move(network)}
    {}
    float fitness = -1.0f;
    Network network;
};

inline void evaluate(Model& model,
                     const std::function<float(const float*, const float*, std::size_t)>& fitness,
                     const std::vector<std::vector<float>>& X,
                     const std::vector<float>& y)
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

inline std::vector<std::unique_ptr<Model>> make_population(const std::size_t size,
                                                           const Network& network,
                                                           const std::function<float(const float*, const float*, std::size_t)>& fitness,
                                                           const std::vector<std::vector<float>>& X,
                                                           const std::vector<float>& y,
                                                           RandomEngine& random_engine)
{
    std::vector<std::unique_ptr<Model>> population;
    population.reserve(size);
    for (std::size_t p = 0; p < size; ++p)
    {
        auto model = std::make_unique<Model>(network);
        model->network.init_weights(random_engine);
        evaluate(*model, fitness, X, y);
        population.emplace_back(std::move(model));
    }
    return population;
}

inline void sort_by_fittest(std::vector<std::unique_ptr<Model>>& population)
{
    std::sort(population.begin(), population.end(), [](const auto& x, const auto& y)
    {
        return x->fitness < y->fitness;
    });
}

inline std::shared_ptr<tw::task<void>> reproduce(std::vector<std::unique_ptr<Model>>& population,
                                                 const std::size_t n_fittest,
                                                 const float crossover_ratio,
                                                 const float mutate_ratio,
                                                 const float mutate_sigma,
                                                 const std::function<float(const float*, const float*, std::size_t)>& fitness,
                                                 const std::vector<std::vector<float>>& X,
                                                 const std::vector<float>& y,
                                                 RandomEngine& random_engine)
{
    assert(n_fittest % 2 == 0);
    std::vector<std::shared_ptr<transwarp::task<void>>> tasks;
    tasks.reserve(n_fittest / 2);
    for (std::size_t i = 0; i < n_fittest; i += 2)
    {
        auto index_task = tw::make_value_task(i);
        auto children_task = tw::make_task(tw::consume,
            [&population, n_fittest, crossover_ratio, mutate_ratio,
             mutate_sigma, &fitness, &X, &y, &random_engine](const auto i)
            {
                auto child1 = std::make_unique<Model>(*population[i]);
                auto child2 = std::make_unique<Model>(*population[i + 1]);
                crossover(child1->network.get_weights().data(),
                          child2->network.get_weights().data(),
                          child1->network.get_weights().size(),
                          crossover_ratio,
                          random_engine);
                mutate(child1->network.get_weights().data(),
                       child1->network.get_weights().size(),
                       mutate_ratio,
                       mutate_sigma,
                       random_engine);
                mutate(child2->network.get_weights().data(),
                       child2->network.get_weights().size(),
                       mutate_ratio,
                       mutate_sigma,
                       random_engine);
                evaluate(*child1, fitness, X, y);
                evaluate(*child2, fitness, X, y);
                population[n_fittest + i] = std::move(child1);
                population[n_fittest + i + 1] = std::move(child2);
            }, index_task);
        tasks.push_back(children_task);
    }
    return transwarp::make_task(transwarp::wait, transwarp::no_op, tasks);
}

inline Model optimize(const Network& network,
                      const std::size_t n_generations,
                      std::size_t population_size,
                      const float crossover_ratio,
                      const float mutate_ratio,
                      const float mutate_sigma,
                      const std::vector<std::vector<float>>& X,
                      const std::vector<std::vector<float>>& y,
                      const std::function<float(const float*, const float*, std::size_t)>& fitness,
                      Printer& printer,
                      RandomEngine& random_engine)
{
    while (population_size % 2 != 0 || (population_size / 2) % 2 != 0)
    {
        // ensure population_size and n_fittest are even
        ++population_size;
    }
    printer("Running GA with population: " + std::to_string(population_size) + "\n");

    const auto y_ref = flatten(y);
    auto population = mlpga::make_population(population_size,
                                             network,
                                             fitness,
                                             X, y_ref, random_engine);
    const auto n_fittest = population_size / 2;
    printer("No of fittest: " + std::to_string(n_fittest) + "\n");

    auto reproduce_task = reproduce(population,
                                    n_fittest,
                                    crossover_ratio,
                                    mutate_ratio,
                                    mutate_sigma,
                                    fitness,
                                    X, y_ref, random_engine);
    const auto n_threads = std::thread::hardware_concurrency();
    printer("No of CPU threads: " + std::to_string(n_threads) + "\n");
    tw::parallel exec{n_threads};

    for (std::size_t g = 0; g < n_generations; ++g)
    {
        printer("Generation: " + std::to_string(g) + "\n");
        reproduce_task->schedule_all(exec);
        reproduce_task->wait();
        mlpga::sort_by_fittest(population);
        printer("Best fitness: " + std::to_string(population.front()->fitness) + "\n");
    }
    return std::move(*population.front());
}

}

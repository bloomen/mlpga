#pragma once

#include <ctime>

#define TRANSWARP_MINIMUM_TASK_SIZE
#define TRANSWARP_CPP11
#include "transwarp.h"
namespace tw = transwarp;

#ifdef MLPGA_USE_GPU
#include "gpufunc.h"
#endif
#include "init.h"
#include "Network.h"
#include "utils.h"

namespace mlpga
{

struct Model
{
    explicit
    Model(Network network = {})
        : network{std::move(network)}
    {}
    float fitness = -1.0f;
    Network network;
};

namespace detail
{

template<typename RandomEngine>
inline void give_birth(float* const w1,
                       float* const w2,
                       const std::size_t n,
                       const float crossover_ratio,
                       const float mutate_ratio,
                       const float mutate_scale,
                       RandomEngine& random_engine)
{
    assert(w1 != nullptr);
    assert(w2 != nullptr);
    assert(n > 0);
    assert(crossover_ratio > 0.0f);
    assert(crossover_ratio < 1.0f);
    assert(mutate_ratio > 0.0f);
    assert(mutate_ratio < 1.0f);
    assert(mutate_scale > 0.0f);
    std::uniform_real_distribution<float> uniform;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (uniform(random_engine) < crossover_ratio)
        {
            std::swap(w1[i], w2[i]);
        }
        if (uniform(random_engine) < mutate_ratio)
        {
            w1[i] += w1[i] * (uniform(random_engine) - 0.5f) * mutate_scale;
        }
        if (uniform(random_engine) < mutate_ratio)
        {
            w2[i] += w2[i] * (uniform(random_engine) - 0.5f) * mutate_scale;
        }
    }
}

inline void evaluate(Model& model,
                     const std::function<float(const float*, const float*, std::size_t)>& fitness,
                     const std::vector<std::vector<float>>& X,
                     const std::vector<float>& y,
                     float* const pred)
{
    const auto n_outputs = model.network.get_layers().back();
    auto p = pred;
    for (const auto& row : X)
    {
        model.network.predict(p, row.data());
        p += n_outputs;
    }
    model.fitness = fitness(y.data(), pred, y.size());
}

template<typename RandomEngine>
inline std::vector<std::unique_ptr<Model>> make_population(const std::size_t size,
                                                           const Network& network,
                                                           const std::function<float(const float*, const float*, std::size_t)>& fitness,
                                                           const std::vector<std::vector<float>>& X,
                                                           const std::vector<float>& y,
                                                           RandomEngine& random_engine)
{
    std::vector<float> pred(y.size());
    std::vector<std::unique_ptr<Model>> population;
    population.reserve(size);
    for (std::size_t p = 0; p < size; ++p)
    {
        auto model = std::unique_ptr<Model>{new Model{network}};
        model->network.init_weights(random_engine);
        evaluate(*model, fitness, X, y, pred.data());
        population.emplace_back(std::move(model));
    }
    return population;
}

inline void sort_by_fittest(std::vector<std::unique_ptr<Model>>& population)
{
    std::sort(population.begin(), population.end(), [](const std::unique_ptr<Model>& x, const std::unique_ptr<Model>& y)
    {
        return x->fitness < y->fitness;
    });
}

struct ReproParams
{
    const std::vector<std::unique_ptr<Model>>& population;
    const std::size_t n_fittest;
    const float crossover_ratio;
    const float mutate_ratio;
    const float mutate_scale;
    const std::function<float(const float*, const float*, std::size_t)>& fitness;
    const std::vector<std::vector<float>>& X;
    const std::vector<float>& y;
};

inline std::shared_ptr<tw::task<void>> reproduce(const ReproParams& params)
{
    assert(params.n_fittest % 2 == 0);
    std::vector<std::shared_ptr<transwarp::task<void>>> tasks;
    tasks.reserve(params.n_fittest / 2);
    for (std::size_t i = 0; i < params.n_fittest; i += 2)
    {
        auto children_task = tw::make_task(tw::root,
            [&params, i]
            {
                // TODO: move this to the GPU. Weight update and predict
                static thread_local std::default_random_engine random_engine{time_seed()};
                auto child1 = params.population[params.n_fittest + i].get();
                auto child2 = params.population[params.n_fittest + i + 1].get();
                *child1 = *params.population[i];
                *child2 = *params.population[i + 1];
                give_birth(child1->network.get_weights().data(),
                           child2->network.get_weights().data(),
                           child1->network.get_weights().size(),
                           params.crossover_ratio,
                           params.mutate_ratio,
                           params.mutate_scale,
                           random_engine);
                static thread_local std::vector<float> pred(params.y.size());
                evaluate(*child1, params.fitness, params.X, params.y, pred.data());
                evaluate(*child2, params.fitness, params.X, params.y, pred.data());
            });
        tasks.push_back(children_task);
    }
    return transwarp::make_task(transwarp::wait, transwarp::no_op, tasks);
}

}

inline Model optimize(const Network& network,
                      const std::size_t n_generations,
                      std::size_t population_size,
                      const float crossover_ratio,
                      const float mutate_ratio,
                      const float mutate_scale,
                      const std::vector<std::vector<float>>& X,
                      const std::vector<std::vector<float>>& y,
                      const std::function<float(const float*, const float*, std::size_t)>& fitness,
                      Printer& printer)
{
    while (population_size % 2 != 0 || (population_size / 2) % 2 != 0)
    {
        // ensure population_size and n_fittest are even
        ++population_size;
    }
    printer("Running GA with population: " + std::to_string(population_size) + "\n");

    std::default_random_engine random_engine{time_seed()};
    const auto y_ref = flatten(y);
    auto population = detail::make_population(population_size,
                                              network,
                                              fitness,
                                              X, y_ref,
                                              random_engine);

    const auto n_fittest = population_size / 2;
    printer("No of fittest: " + std::to_string(n_fittest) + "\n");

    const detail::ReproParams repro_params {
        population,
        n_fittest,
        crossover_ratio,
        mutate_ratio,
        mutate_scale,
        fitness,
        X,
        y_ref
    };
    auto reproduce_task = detail::reproduce(repro_params);

    const auto n_threads = std::thread::hardware_concurrency();
    printer("No of CPU threads: " + std::to_string(n_threads) + "\n");
    tw::parallel exec{n_threads};

    for (std::size_t g = 0; g < n_generations; ++g)
    {
        printer("Generation: " + std::to_string(g) + "\n");
        reproduce_task->schedule_all(exec);
        reproduce_task->wait();
        detail::sort_by_fittest(population);
        printer("Best fitness: " + std::to_string(population.front()->fitness) + "\n");
    }
    return std::move(*population.front());
}

}

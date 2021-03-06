#include <fstream>
#include <iostream>
#include <sstream>

#include "Network.h"
#include "genetic.h"
#include "utils.h"

int main()
{
    std::ifstream f{"/home/cblume/workspace/mlpga/data/boston.csv"};
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    while (!f.eof())
    {
        X.emplace_back(13);
        y.emplace_back(1);
        for (std::size_t i = 0; i < 13; ++i)
        {
            f >> X.back()[i];
        }
        f >> y.back()[0];
    }

    const mlpga::Target target{mlpga::Target::Regression};

    if (target.type == mlpga::Target::Classification)
    {
        for (auto& value : y)
        {
            if (value[0] > 22)
            {
                value[0] = 1;
            }
            else
            {
                value[0] = 0;
            }
        }
    }

    std::default_random_engine random_engine{mlpga::time_seed()};

    const auto split = mlpga::split_train_test(X, y, 0.3f, random_engine);
    std::cout << "Training with " << split.X_train.size() << " samples" << std::endl;

    const std::vector<std::size_t> layers = {13, 13, 1};
    const std::size_t n_generations = 100;
    const std::size_t population_size = 1000;
    const float crossover_ratio = 0.5f;
    const float mutate_ratio = 0.1f;
    const float mutate_scale = 1.0f;

    mlpga::Printer printer = [](const std::string& value) { std::cout << value << std::flush; };
    const mlpga::Network network{target, layers, &random_engine};
    std::cout << network.arch_string() << std::endl;
    const auto model = mlpga::optimize(network, n_generations,
                                       population_size, crossover_ratio,
                                       mutate_ratio, mutate_scale,
                                       split.X_train, split.y_train,
                                       mlpga::mae,
                                       printer);

    std::stringstream ss;
    mlpga::Writer writer = [&ss](const char* v, std::size_t s){ ss.write(v, s); };
    model.network.save(writer);

    ss.seekg(0);
    mlpga::Reader reader = [&ss](char* v, std::size_t s){ ss.read(v, s); };
    const auto net = mlpga::Network::load(reader);

    std::cout << "Prediction..." << std::endl;
    std::vector<std::vector<float>> pred;
    for (const auto& row : split.X_test)
    {
        std::vector<float> output(net.get_layers().back());
        net.predict(output.data(), row.data());
        pred.push_back(std::move(output));
    }

    const auto y_ref = mlpga::flatten(split.y_test);
    std::cout << "MAE=" << mlpga::mae(y_ref.data(), mlpga::flatten(pred).data(), y_ref.size()) << std::endl;
}

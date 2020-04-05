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

    const auto target_type = mlpga::Regression;

    if (target_type == mlpga::Classification)
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

    mlpga::DefaultRandomEngine random_engine{42};

    const auto split = mlpga::split_train_test(X, y, 0.3f, random_engine);
    std::cout << "training with " << split.X_train.size() << " samples" << std::endl;

    const std::vector<std::size_t> layers = {13, 13, 1};
    const std::size_t n_generations = 100;
    const std::size_t population_size = 20;
    const float crossover_ratio = 0.5f;
    const float mutate_ratio = 0.1f;
    const float mutate_sigma = 1.0f;

    mlpga::Printer printer = [](const std::string& value) { std::cout << value << std::flush; };
    const mlpga::Network network{target_type, layers, &random_engine};
    const auto model = mlpga::optimize(network, n_generations,
                                       population_size, crossover_ratio,
                                       mutate_ratio, mutate_sigma,
                                       split.X_train, split.y_train,
                                       mlpga::mae,
                                       printer,
                                       random_engine);

    std::stringstream ss;
    mlpga::Writer writer = [&ss](auto&&... p){ ss.write(std::forward<decltype(p)>(p)...); };
    model.network.save(writer);

    ss.seekg(0);
    mlpga::Reader reader = [&ss](auto&&... p){ ss.read(std::forward<decltype(p)>(p)...); };
    const auto net = mlpga::Network::load(reader);

    std::cout << "prediction" << std::endl;
    std::vector<std::vector<float>> pred;
    for (const auto& row : split.X_test)
    {
        pred.push_back(net.predict(row));
    }

    const auto y_ref = mlpga::flatten(split.y_test);
    std::cout << "MAE=" << mlpga::mae(y_ref.data(), mlpga::flatten(pred).data(), y_ref.size()) << std::endl;
}

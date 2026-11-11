#include <doctest/doctest.h>

#include <numeric>

#include "../envs/envs.hpp"
#include "neat/population.hpp"

using namespace neat;

TEST_CASE("offspring allocation always sums to the population size") {
    for (int total : {1, 7, 150, 151}) {
        const std::vector<double> shares = {0.3, 1.7, 0.0, 5.0, 2.2};
        const auto counts = Population::allocate_offspring(shares, total);
        CHECK(std::accumulate(counts.begin(), counts.end(), 0) == total);
    }
    const auto counts = Population::allocate_offspring({1.0, 3.0}, 100);
    CHECK(counts[0] == 25);
    CHECK(counts[1] == 75);
    const auto equal = Population::allocate_offspring({0.0, 0.0, 0.0}, 9);
    CHECK(equal == std::vector<int>{3, 3, 3});
}

namespace {
Config xor_config(std::uint64_t seed) {
    Config cfg;
    cfg.num_inputs = 2;
    cfg.num_outputs = 1;
    cfg.seed = seed;
    cfg.threads = 2;
    return cfg;
}
Evaluator xor_eval(const Config& cfg) {
    return [cfg](const Genome& g, Rng&) { return envs::xor_fitness(g, cfg); };
}
}  // namespace

TEST_CASE("population size stays constant and every genome gets a species") {
    Config cfg = xor_config(3);
    Population pop(cfg);
    for (int gen = 0; gen < 15; ++gen) {
        pop.step(xor_eval(cfg));
        CHECK(static_cast<int>(pop.genomes().size()) == cfg.population_size);
        int members = 0;
        for (const auto& s : pop.species()) members += static_cast<int>(s.members.size());
        CHECK(members <= cfg.population_size);
        for (const auto& g : pop.genomes()) CHECK(g.species >= 0);
    }
}

TEST_CASE("runs are deterministic for a seed, regardless of thread count") {
    Config a = xor_config(11), b = xor_config(11), c = xor_config(12);
    b.threads = 1;
    Population pa(a), pb(b), pc(c);
    const auto ha = pa.run(xor_eval(a), 20);
    const auto hb = pb.run(xor_eval(b), 20);
    const auto hc = pc.run(xor_eval(c), 20);
    bool differs = false;
    for (std::size_t i = 0; i < ha.size(); ++i) {
        CHECK(ha[i].best_fitness == hb[i].best_fitness);
        CHECK(ha[i].num_species == hb[i].num_species);
        differs = differs || ha[i].best_fitness != hc[i].best_fitness;
    }
    CHECK(differs);
}

TEST_CASE("NEAT solves XOR") {
    int solved = 0;
    for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        Config cfg = xor_config(seed);
        Population pop(cfg);
        pop.run(xor_eval(cfg), 200, [](const GenerationStats& s) { return s.best_fitness >= 15.9; });
        solved += envs::xor_solved(pop.best_ever(), cfg);
    }
    CHECK(solved >= 4);
}

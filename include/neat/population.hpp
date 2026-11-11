#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "neat/genome.hpp"

namespace neat {

struct Species {
    int id = 0;
    Genome representative;
    std::vector<int> members;  // indices into Population::genomes()
    double best_fitness = -1e300;
    int last_improved = 0;
    int created = 0;
};

struct GenerationStats {
    int generation = 0;
    double best_fitness = 0.0;
    double mean_fitness = 0.0;
    int num_species = 0;
    double mean_hidden = 0.0;
    double mean_connections = 0.0;
    double compatibility_threshold = 0.0;
    int best_hidden = 0;
    int best_connections = 0;
};

// Evaluates one genome. Must be thread-safe; `rng` is a private, deterministic stream for this evaluation.
using Evaluator = std::function<double(const Genome&, Rng& rng)>;

class Population {
public:
    explicit Population(const Config& cfg);

    // Evaluates every genome, speciates, records statistics and breeds the next generation.
    GenerationStats step(const Evaluator& evaluate);

    // Runs until `generations` are done or `stop(stats)` returns true. `on_generation` sees each generation's
    // stats while the evaluated population (and its species) is still available.
    std::vector<GenerationStats> run(const Evaluator& evaluate, int generations,
                                     const std::function<bool(const GenerationStats&)>& stop = nullptr,
                                     const std::function<void(const GenerationStats&, const Population&)>& on_generation = nullptr);

    const std::vector<Genome>& genomes() const { return evaluated_; }
    const std::vector<Species>& species() const { return species_; }
    const Genome& best_ever() const { return *best_ever_; }
    const Config& config() const { return cfg_; }
    int generation() const { return generation_; }

    // Exposed for tests.
    void evaluate_all(const Evaluator& evaluate);
    void speciate();
    std::vector<Genome> reproduce();
    static std::vector<int> allocate_offspring(const std::vector<double>& shares, int total);

private:
    Config cfg_;
    Rng rng_;
    InnovationTracker tracker_;
    std::vector<Genome> genomes_;    // population to evaluate next
    std::vector<Genome> evaluated_;  // last evaluated population (with fitness and species)
    std::vector<Species> species_;
    std::optional<Genome> best_ever_;
    int generation_ = 0;
    int next_species_id_ = 0;
};

}  // namespace neat

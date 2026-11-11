#pragma once

#include <cstdint>
#include <string>

namespace neat {

enum class Activation { Sigmoid, Tanh, ReLU, Identity };

std::string to_string(Activation a);
Activation activation_from_string(const std::string& s);

// All NEAT hyperparameters. Defaults follow Stanley & Miikkulainen (2002) where the paper gives values.
struct Config {
    // Network shape
    int num_inputs = 2;
    int num_outputs = 1;
    bool feed_forward = true;  // false: recurrent links allowed, one propagation step per activate()
    Activation hidden_activation = Activation::Sigmoid;
    Activation output_activation = Activation::Sigmoid;
    double sigmoid_steepness = 4.9;  // paper: 1 / (1 + exp(-4.9 x))

    // Population
    int population_size = 150;
    std::uint64_t seed = 1;
    int threads = 0;  // 0 = hardware concurrency

    // Initial weights / weight mutation
    double init_weight_range = 1.0;           // uniform(-r, r)
    double weight_mutate_prob = 0.8;          // chance a genome's weights are mutated
    double weight_perturb_prob = 0.9;         // per connection: perturb (else replace)
    double weight_perturb_sigma = 1.0;
    double weight_replace_range = 2.0;
    double weight_limit = 8.0;

    // Structural mutation
    double add_node_prob = 0.03;
    double add_connection_prob = 0.05;
    double toggle_enable_prob = 0.01;
    int add_connection_attempts = 20;

    // Reproduction
    double crossover_prob = 0.75;             // otherwise: mutation of a single parent
    double interspecies_mating_prob = 0.001;
    double disabled_gene_inherit_prob = 0.75; // gene disabled in either parent stays disabled
    double survival_threshold = 0.2;          // fraction of each species allowed to reproduce
    int elitism_min_species_size = 5;         // champion copied unchanged if species is at least this big

    // Speciation
    double c_excess = 1.0;
    double c_disjoint = 1.0;
    double c_weight = 0.4;
    double compatibility_threshold = 3.0;
    int normalize_min_genes = 20;   // N = 1 for genomes smaller than this (paper)
    int target_species = 0;         // >0: adapt the threshold towards this many species
    double threshold_step = 0.1;
    double threshold_min = 0.3;

    // Stagnation
    int stagnation_generations = 15;
    int protected_top_species = 2;

    // Loads "key = value" lines (# comments). Unknown keys throw.
    static Config from_file(const std::string& path);
    void set(const std::string& key, const std::string& value);
    std::string to_string() const;
};

}  // namespace neat

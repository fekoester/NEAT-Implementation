#pragma once

#include <vector>

#include "neat/genome.hpp"

namespace neat {

double activate(Activation a, double x, double sigmoid_steepness);

// Phenotype built from a genome. Only enabled connections carry signal.
//
// Feed-forward genomes are evaluated in topological order: one activate() call propagates the input all the
// way to the outputs. Recurrent genomes advance one synchronous step per call (node values persist between
// calls until reset()).
class Network {
public:
    static Network from_genome(const Genome& genome, const Config& cfg);

    std::vector<double> activate(const std::vector<double>& inputs);
    void reset();

    int num_nodes() const { return static_cast<int>(values_.size()); }

private:
    struct Link {
        int from;  // dense node index
        double weight;
    };
    struct Neuron {
        int index;
        Activation activation;
        std::vector<Link> incoming;
    };

    bool feed_forward_ = true;
    double steepness_ = 4.9;
    std::vector<int> input_index_;
    int bias_index_ = -1;
    std::vector<int> output_index_;
    std::vector<Neuron> order_;  // evaluation order (topological for feed-forward)
    std::vector<double> values_;
    std::vector<double> scratch_;
};

}  // namespace neat

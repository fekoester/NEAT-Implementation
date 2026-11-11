#pragma once

#include <map>
#include <utility>
#include <vector>

#include "neat/config.hpp"
#include "neat/rng.hpp"

namespace neat {

enum class NodeType { Input, Bias, Output, Hidden };

struct NodeGene {
    int id = 0;
    NodeType type = NodeType::Hidden;
    Activation activation = Activation::Sigmoid;
};

struct ConnectionGene {
    int in = 0;
    int out = 0;
    double weight = 0.0;
    bool enabled = true;
    int innovation = 0;
};

// Global historical markings. The same structural mutation (a link between the same two nodes, or a split
// of the same connection) gets the same innovation number / node id, so genomes can be aligned in crossover
// and compared for speciation.
class InnovationTracker {
public:
    explicit InnovationTracker(int first_free_node_id) : next_node_id_(first_free_node_id) {}

    int connection_innovation(int in, int out);

    // Node id for splitting connection `innovation`. Reused within a generation so identical splits match;
    // `taken(id)` lets a genome that already owns that node (after re-enabling and re-splitting) get a fresh id.
    template <class Taken>
    int split_node_id(int innovation, Taken taken) {
        auto it = splits_.find(innovation);
        if (it != splits_.end() && !taken(it->second)) return it->second;
        const int id = next_node_id_++;
        splits_[innovation] = id;
        return id;
    }

    // Clears per-generation split bookkeeping (connection innovations persist).
    void new_generation() { splits_.clear(); }

    int next_node_id() const { return next_node_id_; }
    int next_innovation() const { return next_innovation_; }

private:
    std::map<std::pair<int, int>, int> connections_;
    std::map<int, int> splits_;
    int next_innovation_ = 0;
    int next_node_id_;
};

class Genome {
public:
    std::vector<NodeGene> nodes;        // sorted by id
    std::vector<ConnectionGene> conns;  // sorted by innovation
    double fitness = 0.0;
    double adjusted_fitness = 0.0;
    int species = -1;

    // Fully connected inputs + bias -> outputs, no hidden nodes (NEAT's minimal start).
    static Genome minimal(const Config& cfg, InnovationTracker& tracker, Rng& rng);

    // Node ids: inputs 0..n_in-1, bias n_in, outputs n_in+1..n_in+n_out.
    static int bias_id(const Config& cfg) { return cfg.num_inputs; }
    static int first_output_id(const Config& cfg) { return cfg.num_inputs + 1; }
    static int first_hidden_id(const Config& cfg) { return cfg.num_inputs + 1 + cfg.num_outputs; }

    const NodeGene* find_node(int id) const;
    bool has_node(int id) const { return find_node(id) != nullptr; }
    bool has_connection(int in, int out) const;
    int num_hidden() const;
    int num_enabled() const;

    // Mutations
    void mutate(const Config& cfg, InnovationTracker& tracker, Rng& rng);
    void mutate_weights(const Config& cfg, Rng& rng);
    bool mutate_add_node(const Config& cfg, InnovationTracker& tracker, Rng& rng);
    bool mutate_add_connection(const Config& cfg, InnovationTracker& tracker, Rng& rng);
    void mutate_toggle_enable(Rng& rng);

    // `fitter` contributes all disjoint/excess genes; matching genes are chosen at random.
    static Genome crossover(const Genome& fitter, const Genome& other, const Config& cfg, Rng& rng);

    // delta = c1*E/N + c2*D/N + c3*W (Stanley & Miikkulainen 2002, eq. 1)
    static double compatibility(const Genome& a, const Genome& b, const Config& cfg);

    // True if adding in->out would create a cycle (for feed-forward networks).
    bool creates_cycle(int in, int out) const;

    void sort_genes();

private:
    void add_node(NodeGene node);
};

}  // namespace neat

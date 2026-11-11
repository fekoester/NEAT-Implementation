#include "neat/genome.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace neat {

int InnovationTracker::connection_innovation(int in, int out) {
    const auto key = std::make_pair(in, out);
    auto it = connections_.find(key);
    if (it != connections_.end()) return it->second;
    const int innovation = next_innovation_++;
    connections_.emplace(key, innovation);
    return innovation;
}

Genome Genome::minimal(const Config& cfg, InnovationTracker& tracker, Rng& rng) {
    Genome g;
    for (int i = 0; i < cfg.num_inputs; ++i) g.nodes.push_back({i, NodeType::Input, Activation::Identity});
    g.nodes.push_back({bias_id(cfg), NodeType::Bias, Activation::Identity});
    for (int o = 0; o < cfg.num_outputs; ++o)
        g.nodes.push_back({first_output_id(cfg) + o, NodeType::Output, cfg.output_activation});

    for (int o = 0; o < cfg.num_outputs; ++o) {
        const int out = first_output_id(cfg) + o;
        for (int in = 0; in <= cfg.num_inputs; ++in) {  // inputs + bias
            ConnectionGene c;
            c.in = in;
            c.out = out;
            c.weight = rng.uniform(-cfg.init_weight_range, cfg.init_weight_range);
            c.innovation = tracker.connection_innovation(in, out);
            g.conns.push_back(c);
        }
    }
    g.sort_genes();
    return g;
}

const NodeGene* Genome::find_node(int id) const {
    auto it = std::lower_bound(nodes.begin(), nodes.end(), id,
                               [](const NodeGene& n, int v) { return n.id < v; });
    return (it != nodes.end() && it->id == id) ? &*it : nullptr;
}

bool Genome::has_connection(int in, int out) const {
    return std::any_of(conns.begin(), conns.end(), [&](const ConnectionGene& c) { return c.in == in && c.out == out; });
}

int Genome::num_hidden() const {
    return static_cast<int>(std::count_if(nodes.begin(), nodes.end(), [](const NodeGene& n) { return n.type == NodeType::Hidden; }));
}

int Genome::num_enabled() const {
    return static_cast<int>(std::count_if(conns.begin(), conns.end(), [](const ConnectionGene& c) { return c.enabled; }));
}

void Genome::sort_genes() {
    std::sort(nodes.begin(), nodes.end(), [](const NodeGene& a, const NodeGene& b) { return a.id < b.id; });
    std::sort(conns.begin(), conns.end(), [](const ConnectionGene& a, const ConnectionGene& b) { return a.innovation < b.innovation; });
}

void Genome::add_node(NodeGene node) {
    auto it = std::lower_bound(nodes.begin(), nodes.end(), node.id,
                               [](const NodeGene& n, int v) { return n.id < v; });
    nodes.insert(it, node);
}

void Genome::mutate(const Config& cfg, InnovationTracker& tracker, Rng& rng) {
    if (rng.chance(cfg.add_node_prob)) mutate_add_node(cfg, tracker, rng);
    if (rng.chance(cfg.add_connection_prob)) mutate_add_connection(cfg, tracker, rng);
    if (rng.chance(cfg.weight_mutate_prob)) mutate_weights(cfg, rng);
    if (rng.chance(cfg.toggle_enable_prob)) mutate_toggle_enable(rng);
}

void Genome::mutate_weights(const Config& cfg, Rng& rng) {
    for (auto& c : conns) {
        if (rng.chance(cfg.weight_perturb_prob)) {
            c.weight += rng.normal(0.0, cfg.weight_perturb_sigma);
        } else {
            c.weight = rng.uniform(-cfg.weight_replace_range, cfg.weight_replace_range);
        }
        c.weight = std::clamp(c.weight, -cfg.weight_limit, cfg.weight_limit);
    }
}

bool Genome::mutate_add_node(const Config& cfg, InnovationTracker& tracker, Rng& rng) {
    std::vector<int> enabled;
    for (int i = 0; i < static_cast<int>(conns.size()); ++i)
        if (conns[i].enabled) enabled.push_back(i);
    if (enabled.empty()) return false;

    ConnectionGene& split = conns[enabled[rng.index(enabled.size())]];
    split.enabled = false;
    const int in = split.in, out = split.out;
    const double weight = split.weight;
    const int node_id = tracker.split_node_id(split.innovation, [this](int id) { return has_node(id); });

    add_node({node_id, NodeType::Hidden, cfg.hidden_activation});
    // in -> new with weight 1 and new -> out with the old weight: the network's function barely changes.
    conns.push_back({in, node_id, 1.0, true, tracker.connection_innovation(in, node_id)});
    conns.push_back({node_id, out, weight, true, tracker.connection_innovation(node_id, out)});
    sort_genes();
    return true;
}

bool Genome::creates_cycle(int in, int out) const {
    if (in == out) return true;
    // Is `in` reachable from `out`? (All genes, enabled or not, so re-enabling never creates a cycle.)
    std::unordered_map<int, std::vector<int>> adjacency;
    for (const auto& c : conns) adjacency[c.in].push_back(c.out);
    std::vector<int> stack{out};
    std::unordered_set<int> seen{out};
    while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        if (node == in) return true;
        for (int next : adjacency[node])
            if (seen.insert(next).second) stack.push_back(next);
    }
    return false;
}

bool Genome::mutate_add_connection(const Config& cfg, InnovationTracker& tracker, Rng& rng) {
    std::vector<int> sources, targets;
    for (const auto& n : nodes) {
        if (n.type != NodeType::Output || !cfg.feed_forward) sources.push_back(n.id);
        if (n.type == NodeType::Output || n.type == NodeType::Hidden) targets.push_back(n.id);
    }
    for (int attempt = 0; attempt < cfg.add_connection_attempts; ++attempt) {
        const int in = sources[rng.index(sources.size())];
        const int out = targets[rng.index(targets.size())];
        if (has_connection(in, out)) continue;
        if (cfg.feed_forward && creates_cycle(in, out)) continue;
        const double w = rng.uniform(-cfg.init_weight_range, cfg.init_weight_range);
        conns.push_back({in, out, w, true, tracker.connection_innovation(in, out)});
        sort_genes();
        return true;
    }
    return false;
}

void Genome::mutate_toggle_enable(Rng& rng) {
    if (conns.empty()) return;
    auto& c = conns[rng.index(conns.size())];
    c.enabled = !c.enabled;  // the 2017 code wrote `enabled != enabled`, a no-op comparison
}

Genome Genome::crossover(const Genome& fitter, const Genome& other, const Config& cfg, Rng& rng) {
    std::unordered_map<int, const ConnectionGene*> other_by_innovation;
    for (const auto& c : other.conns) other_by_innovation[c.innovation] = &c;

    Genome child;
    child.nodes = fitter.nodes;  // child has the fitter parent's structure
    child.conns.reserve(fitter.conns.size());
    for (const auto& gene : fitter.conns) {
        auto it = other_by_innovation.find(gene.innovation);
        if (it == other_by_innovation.end()) {
            child.conns.push_back(gene);  // disjoint / excess: from the fitter parent
            continue;
        }
        const ConnectionGene& partner = *it->second;
        ConnectionGene c = rng.chance(0.5) ? gene : partner;  // matching: random parent
        if (!gene.enabled || !partner.enabled)
            c.enabled = !rng.chance(cfg.disabled_gene_inherit_prob);
        else
            c.enabled = true;
        child.conns.push_back(c);
    }
    child.sort_genes();
    return child;
}

double Genome::compatibility(const Genome& a, const Genome& b, const Config& cfg) {
    std::size_t i = 0, j = 0;
    int matching = 0, disjoint = 0, excess = 0;
    double weight_diff = 0.0;
    const int max_a = a.conns.empty() ? -1 : a.conns.back().innovation;
    const int max_b = b.conns.empty() ? -1 : b.conns.back().innovation;

    while (i < a.conns.size() || j < b.conns.size()) {
        if (i < a.conns.size() && j < b.conns.size() && a.conns[i].innovation == b.conns[j].innovation) {
            weight_diff += std::abs(a.conns[i].weight - b.conns[j].weight);
            ++matching, ++i, ++j;
        } else if (j >= b.conns.size() || (i < a.conns.size() && a.conns[i].innovation < b.conns[j].innovation)) {
            (a.conns[i].innovation > max_b ? excess : disjoint) += 1;
            ++i;
        } else {
            (b.conns[j].innovation > max_a ? excess : disjoint) += 1;
            ++j;
        }
    }
    const std::size_t larger = std::max(a.conns.size(), b.conns.size());
    const double n = larger < static_cast<std::size_t>(cfg.normalize_min_genes) ? 1.0 : static_cast<double>(larger);
    const double mean_weight_diff = matching > 0 ? weight_diff / matching : 0.0;
    return cfg.c_excess * excess / n + cfg.c_disjoint * disjoint / n + cfg.c_weight * mean_weight_diff;
}

}  // namespace neat

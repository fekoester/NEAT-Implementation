#include "neat/network.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace neat {

double activate(Activation a, double x, double sigmoid_steepness) {
    switch (a) {
        case Activation::Sigmoid:
            return 1.0 / (1.0 + std::exp(-sigmoid_steepness * x));
        case Activation::Tanh:
            return std::tanh(x);
        case Activation::ReLU:
            return x > 0.0 ? x : 0.0;
        case Activation::Identity:
            return x;
    }
    return x;
}

Network Network::from_genome(const Genome& genome, const Config& cfg) {
    Network net;
    net.feed_forward_ = cfg.feed_forward;
    net.steepness_ = cfg.sigmoid_steepness;

    std::unordered_map<int, int> index;  // node id -> dense index
    for (const auto& n : genome.nodes) {
        const int i = static_cast<int>(index.size());
        index[n.id] = i;
        if (n.type == NodeType::Input) net.input_index_.push_back(i);
        if (n.type == NodeType::Bias) net.bias_index_ = i;
        if (n.type == NodeType::Output) net.output_index_.push_back(i);
    }
    net.values_.assign(index.size(), 0.0);
    net.scratch_.assign(index.size(), 0.0);

    std::unordered_map<int, std::vector<Link>> incoming;
    for (const auto& c : genome.conns) {
        if (!c.enabled) continue;
        incoming[index.at(c.out)].push_back({index.at(c.in), c.weight});
    }

    auto make_neuron = [&](const NodeGene& n) {
        const int i = index.at(n.id);
        return Neuron{i, n.activation, incoming.count(i) ? incoming[i] : std::vector<Link>{}};
    };

    if (!cfg.feed_forward) {
        for (const auto& n : genome.nodes)
            if (n.type == NodeType::Hidden || n.type == NodeType::Output) net.order_.push_back(make_neuron(n));
        return net;
    }

    // Topological order (Kahn) over enabled links. Nodes that cannot reach an output are skipped.
    std::vector<bool> needed(index.size(), false);
    {
        std::vector<int> stack(net.output_index_.begin(), net.output_index_.end());
        for (int o : stack) needed[o] = true;
        while (!stack.empty()) {
            const int node = stack.back();
            stack.pop_back();
            for (const auto& link : incoming[node])
                if (!needed[link.from]) {
                    needed[link.from] = true;
                    stack.push_back(link.from);
                }
        }
    }
    std::vector<int> pending(index.size(), 0);
    std::unordered_map<int, std::vector<int>> outgoing;
    for (const auto& [to, links] : incoming)
        for (const auto& link : links) {
            outgoing[link.from].push_back(to);
            ++pending[to];
        }
    std::queue<int> ready;
    for (const auto& n : genome.nodes)
        if (pending[index.at(n.id)] == 0) ready.push(index.at(n.id));
    std::unordered_map<int, const NodeGene*> by_index;
    for (const auto& n : genome.nodes) by_index[index.at(n.id)] = &n;

    std::size_t visited = 0;
    while (!ready.empty()) {
        const int node = ready.front();
        ready.pop();
        ++visited;
        const NodeGene& gene = *by_index[node];
        if (needed[node] && (gene.type == NodeType::Hidden || gene.type == NodeType::Output))
            net.order_.push_back(make_neuron(gene));
        for (int next : outgoing[node])
            if (--pending[next] == 0) ready.push(next);
    }
    if (visited != index.size()) throw std::runtime_error("feed-forward genome contains a cycle");
    return net;
}

std::vector<double> Network::activate(const std::vector<double>& inputs) {
    if (inputs.size() != input_index_.size()) throw std::invalid_argument("wrong number of network inputs");
    for (std::size_t i = 0; i < inputs.size(); ++i) values_[input_index_[i]] = inputs[i];
    if (bias_index_ >= 0) values_[bias_index_] = 1.0;

    if (feed_forward_) {
        for (const auto& neuron : order_) {
            double sum = 0.0;
            for (const auto& link : neuron.incoming) sum += link.weight * values_[link.from];
            values_[neuron.index] = neat::activate(neuron.activation, sum, steepness_);
        }
    } else {
        // Synchronous update: every neuron reads the previous step's values.
        scratch_ = values_;
        for (const auto& neuron : order_) {
            double sum = 0.0;
            for (const auto& link : neuron.incoming) sum += link.weight * values_[link.from];
            scratch_[neuron.index] = neat::activate(neuron.activation, sum, steepness_);
        }
        std::swap(values_, scratch_);
    }

    std::vector<double> out;
    out.reserve(output_index_.size());
    for (int o : output_index_) out.push_back(values_[o]);
    return out;
}

void Network::reset() { std::fill(values_.begin(), values_.end(), 0.0); }

}  // namespace neat

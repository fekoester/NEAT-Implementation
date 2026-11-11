#include <doctest/doctest.h>

#include <stdexcept>

#include "neat/network.hpp"

using namespace neat;

namespace {
// Hand-built XOR: h1 = OR, h2 = AND, out = h1 AND NOT h2 (steep sigmoids).
Genome xor_genome() {
    Genome g;
    g.nodes = {{0, NodeType::Input, Activation::Identity}, {1, NodeType::Input, Activation::Identity},
               {2, NodeType::Bias, Activation::Identity},  {3, NodeType::Output, Activation::Sigmoid},
               {4, NodeType::Hidden, Activation::Sigmoid}, {5, NodeType::Hidden, Activation::Sigmoid}};
    g.conns = {{0, 4, 4, true, 0},  {1, 4, 4, true, 1},  {2, 4, -2, true, 2},  {0, 5, 4, true, 3},
               {1, 5, 4, true, 4},  {2, 5, -6, true, 5}, {4, 3, 6, true, 6},   {5, 3, -6, true, 7},
               {2, 3, -3, true, 8}};
    return g;
}
}  // namespace

TEST_CASE("feed-forward network computes XOR in one activation") {
    Config cfg;
    auto net = Network::from_genome(xor_genome(), cfg);
    CHECK(net.activate({0, 0})[0] < 0.1);
    CHECK(net.activate({0, 1})[0] > 0.9);
    CHECK(net.activate({1, 0})[0] > 0.9);
    CHECK(net.activate({1, 1})[0] < 0.1);
}

TEST_CASE("disabled connections carry no signal (the 2017 network ignored the flag)") {
    Config cfg;
    Genome g = xor_genome();
    for (auto& c : g.conns) c.enabled = false;
    auto net = Network::from_genome(g, cfg);
    CHECK(net.activate({1, 1})[0] == doctest::Approx(0.5));  // sigmoid(0)
}

TEST_CASE("recurrent networks advance one step per activation") {
    Config cfg;
    cfg.feed_forward = false;
    Genome g;
    g.nodes = {{0, NodeType::Input, Activation::Identity}, {1, NodeType::Bias, Activation::Identity},
               {2, NodeType::Output, Activation::Identity}, {3, NodeType::Hidden, Activation::Identity}};
    g.conns = {{0, 3, 1.0, true, 0}, {3, 2, 1.0, true, 1}, {2, 2, 0.5, true, 2}};
    auto net = Network::from_genome(g, cfg);
    CHECK(net.activate({1})[0] == doctest::Approx(0.0));  // input reaches the hidden node first
    CHECK(net.activate({0})[0] == doctest::Approx(1.0));
    CHECK(net.activate({0})[0] == doctest::Approx(0.5));  // self-loop decays
    net.reset();
    CHECK(net.activate({0})[0] == doctest::Approx(0.0));
}

TEST_CASE("feed-forward construction rejects cycles") {
    Config cfg;
    Genome g = xor_genome();
    g.conns.push_back({3, 4, 1.0, true, 9});
    CHECK_THROWS_AS(Network::from_genome(g, cfg), std::runtime_error);
}

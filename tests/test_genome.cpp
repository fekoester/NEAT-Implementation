#include <doctest/doctest.h>

#include <algorithm>
#include <set>

#include "neat/genome.hpp"

using namespace neat;

namespace {
Config small_config() {
    Config cfg;
    cfg.num_inputs = 3;
    cfg.num_outputs = 2;
    return cfg;
}
}  // namespace

TEST_CASE("minimal genome is fully connected inputs+bias -> outputs") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(1);
    Genome g = Genome::minimal(cfg, tracker, rng);
    CHECK(g.nodes.size() == 3 + 1 + 2);
    CHECK(g.conns.size() == (3 + 1) * 2);
    CHECK(g.num_hidden() == 0);
    Genome h = Genome::minimal(cfg, tracker, rng);
    for (std::size_t i = 0; i < g.conns.size(); ++i) CHECK(g.conns[i].innovation == h.conns[i].innovation);
}

TEST_CASE("add node splits a connection and reuses ids for identical splits in a generation") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(2);
    Genome a = Genome::minimal(cfg, tracker, rng);
    Genome b = a;
    Rng ra(5), rb(5);  // same choice of connection
    REQUIRE(a.mutate_add_node(cfg, tracker, ra));
    REQUIRE(b.mutate_add_node(cfg, tracker, rb));
    CHECK(a.num_hidden() == 1);
    CHECK(a.conns.size() == 8 + 2);
    CHECK(std::count_if(a.conns.begin(), a.conns.end(), [](auto& c) { return !c.enabled; }) == 1);
    CHECK(a.nodes.back().id == b.nodes.back().id);
    for (std::size_t i = 0; i < a.conns.size(); ++i) CHECK(a.conns[i].innovation == b.conns[i].innovation);

    tracker.new_generation();
    Genome c = Genome::minimal(cfg, tracker, rng);
    Rng rc(5);
    REQUIRE(c.mutate_add_node(cfg, tracker, rc));
    CHECK(c.nodes.back().id != a.nodes.back().id);  // new generation: a fresh node id
}

TEST_CASE("node ids stay unique when a re-enabled connection is split again") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(3);
    Genome g = Genome::minimal(cfg, tracker, rng);
    g.conns.resize(1);
    Rng r1(9);
    REQUIRE(g.mutate_add_node(cfg, tracker, r1));
    g.conns.front().enabled = true;  // re-enable the split connection, split it again
    for (auto& c : g.conns) c.enabled = c.innovation == g.conns.front().innovation;
    Rng r2(9);
    REQUIRE(g.mutate_add_node(cfg, tracker, r2));
    std::set<int> ids;
    for (const auto& n : g.nodes) CHECK(ids.insert(n.id).second);
}

TEST_CASE("toggle enable actually flips the flag (was a no-op in 2017)") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(4);
    Genome g = Genome::minimal(cfg, tracker, rng);
    g.conns.resize(1);
    g.mutate_toggle_enable(rng);
    CHECK_FALSE(g.conns[0].enabled);
    g.mutate_toggle_enable(rng);
    CHECK(g.conns[0].enabled);
}

TEST_CASE("feed-forward connection mutations never create cycles") {
    Config cfg = small_config();
    cfg.add_connection_attempts = 50;
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(6);
    Genome g = Genome::minimal(cfg, tracker, rng);
    for (int i = 0; i < 300; ++i) {
        if (rng.chance(0.3)) g.mutate_add_node(cfg, tracker, rng);
        g.mutate_add_connection(cfg, tracker, rng);
    }
    for (const auto& c : g.conns) {
        CHECK(c.in != c.out);
        CHECK(g.find_node(c.out)->type != NodeType::Input);
        CHECK(g.find_node(c.out)->type != NodeType::Bias);
    }
    // No duplicate links.
    std::set<std::pair<int, int>> links;
    for (const auto& c : g.conns) CHECK(links.insert({c.in, c.out}).second);
    // Removing any link from the DAG: adding it back must not be a cycle; adding the reverse must be.
    for (const auto& c : g.conns) {
        Genome without = g;
        without.conns.erase(std::find_if(without.conns.begin(), without.conns.end(),
                                         [&](auto& x) { return x.innovation == c.innovation; }));
        CHECK_FALSE(without.creates_cycle(c.in, c.out));
        CHECK(g.creates_cycle(c.out, c.in));
    }
}

TEST_CASE("crossover aligns genes by innovation") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(7);
    Genome fit = Genome::minimal(cfg, tracker, rng);
    Genome other = Genome::minimal(cfg, tracker, rng);
    fit.mutate_add_node(cfg, tracker, rng);       // disjoint/excess genes in the fitter parent
    other.mutate_add_connection(cfg, tracker, rng);  // and some only in the other one

    int from_other = 0;
    for (int trial = 0; trial < 50; ++trial) {
        Genome child = Genome::crossover(fit, other, cfg, rng);
        REQUIRE(child.conns.size() == fit.conns.size());  // structure of the fitter parent
        for (std::size_t i = 0; i < child.conns.size(); ++i) {
            CHECK(child.conns[i].innovation == fit.conns[i].innovation);
            for (const auto& o : other.conns)
                if (o.innovation == child.conns[i].innovation && o.weight != fit.conns[i].weight &&
                    child.conns[i].weight == o.weight)
                    ++from_other;
        }
    }
    CHECK(from_other > 0);  // matching genes really come from both parents (the 2017 loop never did)
}

TEST_CASE("compatibility distance") {
    Config cfg = small_config();
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(8);
    Genome a = Genome::minimal(cfg, tracker, rng);
    CHECK(Genome::compatibility(a, a, cfg) == doctest::Approx(0.0));

    Genome b = a;
    for (auto& c : b.conns) c.weight += 1.0;
    CHECK(Genome::compatibility(a, b, cfg) == doctest::Approx(cfg.c_weight * 1.0));

    Genome c = a;
    c.mutate_add_node(cfg, tracker, rng);  // 2 excess genes, N = 1 for small genomes
    CHECK(Genome::compatibility(a, c, cfg) == doctest::Approx(2.0 * cfg.c_excess));
    CHECK(Genome::compatibility(c, a, cfg) == doctest::Approx(Genome::compatibility(a, c, cfg)));

    Genome empty;
    CHECK(Genome::compatibility(empty, empty, cfg) == doctest::Approx(0.0));  // no division by zero
}

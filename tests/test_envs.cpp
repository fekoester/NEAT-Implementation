#include <doctest/doctest.h>

#include <sstream>

#include "../envs/envs.hpp"
#include "neat/io.hpp"

using namespace neat;

TEST_CASE("cart-pole falls without control and the physics are deterministic") {
    envs::CartPole a, b;
    a.s.theta = b.s.theta = 0.05;
    int steps = 0;
    while (!a.failed() && steps < 1000) {
        a.step(steps % 2 == 0);
        b.step(steps % 2 == 0);
        ++steps;
    }
    CHECK(steps < 1000);
    CHECK(a.s.theta == b.s.theta);
}

TEST_CASE("dodge game: fixed time step, deterministic, sensible observations") {
    envs::DodgeGame a(42), b(42);
    for (int t = 0; t < 600; ++t) {
        const bool alive_a = a.step(1), alive_b = b.step(1);
        CHECK(alive_a == alive_b);
        if (!alive_a) break;
    }
    CHECK(a.time == b.time);
    const auto obs = a.observation();
    CHECK(obs.size() == 2 * envs::DodgeGame::rows);
    double one_hot = 0;
    for (int r = 0; r < envs::DodgeGame::rows; ++r) {
        CHECK(obs[r] >= 0.0);
        CHECK(obs[r] <= 1.0);
        one_hot += obs[envs::DodgeGame::rows + r];
    }
    CHECK(one_hot == 1.0);
}

TEST_CASE("dodge: standing still eventually gets hit; time is capped") {
    envs::DodgeGame g(1);
    while (g.step(1)) {
    }
    CHECK(g.hit);
    CHECK(g.time < envs::DodgeGame::max_seconds);
}

TEST_CASE("genome save/load round trip and DOT export") {
    Config cfg;
    InnovationTracker tracker(Genome::first_hidden_id(cfg));
    Rng rng(1);
    Genome g = Genome::minimal(cfg, tracker, rng);
    g.mutate_add_node(cfg, tracker, rng);
    g.fitness = 3.25;
    std::stringstream ss;
    save_genome(g, ss);
    Genome h = load_genome(ss);
    REQUIRE(h.conns.size() == g.conns.size());
    REQUIRE(h.nodes.size() == g.nodes.size());
    CHECK(h.fitness == g.fitness);
    for (std::size_t i = 0; i < g.conns.size(); ++i) {
        CHECK(h.conns[i].weight == g.conns[i].weight);
        CHECK(h.conns[i].enabled == g.conns[i].enabled);
    }
    std::stringstream dot;
    write_dot(g, dot, "test");
    CHECK(dot.str().find("digraph") != std::string::npos);
    CHECK(dot.str().find("style=dashed") != std::string::npos);  // the split connection
}

TEST_CASE("config parsing") {
    Config cfg;
    cfg.set("population_size", "77");
    cfg.set("feed_forward", "false");
    cfg.set("hidden_activation", "tanh");
    CHECK(cfg.population_size == 77);
    CHECK_FALSE(cfg.feed_forward);
    CHECK(cfg.hidden_activation == Activation::Tanh);
    CHECK_THROWS(cfg.set("no_such_key", "1"));
    CHECK(cfg.to_string().find("population_size = 77") != std::string::npos);
}

TEST_CASE("double pole: unstable without control, deterministic, fails within bounds") {
    envs::DoublePole a, b;
    int steps = 0;
    while (!a.failed() && steps < 10000) {
        a.step(0.0);
        b.step(0.0);
        ++steps;
    }
    CHECK(steps < 10000);  // the 4.5 degree start falls over without control
    CHECK(a.s.theta1 == b.s.theta1);
    CHECK(a.observation().size() == 6);

    // A strong constant push drives the cart off the track.
    envs::DoublePole c;
    c.s.theta1 = 0.0;
    int push_steps = 0;
    while (!c.failed() && push_steps < 10000) {
        c.step(10.0);
        ++push_steps;
    }
    CHECK(push_steps < 10000);
}

TEST_CASE("environment registry") {
    for (const auto& name : envs::names()) {
        const auto& env = envs::get(name);
        CHECK(env.name == name);
        CHECK(env.num_inputs > 0);
        CHECK(env.num_outputs > 0);
        CHECK(static_cast<bool>(env.fitness));
    }
    CHECK_THROWS(envs::get("no_such_env"));
}

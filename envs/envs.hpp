#pragma once

#include <functional>
#include <string>
#include <vector>

#include "neat/genome.hpp"

namespace neat::envs {

// A task NEAT can be trained on.
struct Environment {
    std::string name;
    std::string description;
    int num_inputs = 0;
    int num_outputs = 0;
    double fitness_threshold = 0.0;  // training stops early once the champion's fitness reaches this
    // Optional task-specific success test for the generation champion (e.g. XOR: all four cases correct,
    // the criterion of the NEAT paper). If set, it is used instead of fitness_threshold.
    std::function<bool(const Genome&, const Config&)> solved;
    // Fitness of a genome (thread-safe; `rng` drives any randomness, e.g. episode seeds).
    std::function<double(const Genome&, const Config&, Rng&)> fitness;
    // Optional validation score used to pick the champion among generation champions (noisy tasks: the
    // genome with the best training fitness is often just the luckiest). Uses its own episode seeds.
    std::function<double(const Genome&, const Config&)> validation;
    // Stricter post-training test of a champion (e.g. unseen episode seeds). Returns a score in [0, 1].
    std::function<double(const Genome&, const Config&)> generalization;
    // Suggested config overrides for this task.
    std::vector<std::pair<std::string, std::string>> defaults;
};

const Environment& get(const std::string& name);
std::vector<std::string> names();

// ---- XOR -------------------------------------------------------------------------------------------
// Fitness (4 - sum|error|)^2 as in Stanley & Miikkulainen (2002); solved when every output rounds
// to the right answer.
double xor_fitness(const Genome& g, const Config& cfg);
bool xor_solved(const Genome& g, const Config& cfg);

// ---- Cart-pole -------------------------------------------------------------------------------------
struct CartPoleState {
    double x = 0, x_dot = 0, theta = 0, theta_dot = 0;
};
struct CartPole {
    static constexpr double gravity = 9.8, cart_mass = 1.0, pole_mass = 0.1, half_length = 0.5;
    static constexpr double force = 10.0, dt = 0.02, x_limit = 2.4, theta_limit = 12.0 * 3.14159265358979 / 180.0;
    CartPoleState s;
    void step(bool push_right);
    bool failed() const;
    std::vector<double> observation() const;
};
// Number of steps the network balances the pole (up to max_steps) from a random start.
int cartpole_episode(const Genome& g, const Config& cfg, Rng& rng, int max_steps, double start_range);

// ---- Double-pole balancing with velocities (the control benchmark of the NEAT paper) ------------------
// Two poles (half-lengths 0.5 m and 0.05 m) on one cart, Wieland's equations, RK4 with dt = 0.01 s and two
// integration steps per decision, failure at |x| > 2.4 m or either pole beyond 36 degrees. Constants as in the
// classic NEAT / ESP experiments. Solved = 100,000 decision steps (about 33 minutes of simulated time).
struct DoublePoleState {
    double x = 0, x_dot = 0, theta1 = 0.07853981633974483 /* 4.5 deg */, theta1_dot = 0, theta2 = 0, theta2_dot = 0;
};
struct DoublePole {
    DoublePoleState s;
    void step(double force);  // force in [-10, 10] N
    bool failed() const;
    std::vector<double> observation() const;
};
int double_pole_episode(const Genome& g, const Config& cfg, const DoublePoleState& start, int max_steps);

// ---- Dodge (the 2017 game, deterministic) -----------------------------------------------------------
// The player sits at the right edge and switches between rows; projectiles fly in from the left at random
// speeds, spawning faster over time. Fixed 60 Hz time step, so fitness no longer depends on machine speed.
struct DodgeFrame {
    int player_row;
    std::vector<std::pair<double, int>> projectiles;  // (x, row)
};
struct DodgeGame {
    static constexpr int rows = 6;
    static constexpr double width = 720.0, dt = 1.0 / 60.0, player_x = width - 21.0, hit_distance = 14.0;
    static constexpr double min_speed = 150.0, max_speed = 350.0, move_cooldown = 0.12;
    static constexpr double max_seconds = 60.0;

    struct Projectile {
        double x;
        int row;
        double speed;
    };

    explicit DodgeGame(std::uint64_t seed);
    // Inputs: per row, proximity (0 = none .. 1 = impact) of the nearest incoming projectile;
    // then the player's row as one-hot.
    std::vector<double> observation() const;
    // action: 0 = up, 1 = stay, 2 = down. Returns false once the player is hit or time runs out.
    bool step(int action);

    double time = 0.0;
    int player_row = rows / 2;
    bool hit = false;
    std::vector<Projectile> projectiles;
    DodgeFrame frame() const;

private:
    Rng rng_;
    double spawn_timer_ = 0.0;
    double cooldown_ = 0.0;
    double spawn_interval() const;
};
double dodge_episode(const Genome& g, const Config& cfg, std::uint64_t seed, std::vector<DodgeFrame>* trace = nullptr);

}  // namespace neat::envs

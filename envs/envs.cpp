#include "envs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "neat/network.hpp"

namespace neat::envs {

// ---------------------------------------------------------------------------------------------------- XOR

double xor_fitness(const Genome& g, const Config& cfg) {
    static const double cases[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    auto net = Network::from_genome(g, cfg);
    double error = 0.0;
    for (const auto& c : cases) {
        net.reset();
        error += std::abs(net.activate({c[0], c[1]})[0] - c[2]);
    }
    return (4.0 - error) * (4.0 - error);
}

bool xor_solved(const Genome& g, const Config& cfg) {
    static const double cases[4][3] = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    auto net = Network::from_genome(g, cfg);
    for (const auto& c : cases) {
        net.reset();
        if (std::abs(net.activate({c[0], c[1]})[0] - c[2]) >= 0.5) return false;
    }
    return true;
}

// ----------------------------------------------------------------------------------------------- Cart-pole

void CartPole::step(bool push_right) {
    // Barto, Sutton & Anderson (1983) dynamics, Euler integration.
    const double f = push_right ? force : -force;
    const double total_mass = cart_mass + pole_mass;
    const double cos_t = std::cos(s.theta), sin_t = std::sin(s.theta);
    const double temp = (f + pole_mass * half_length * s.theta_dot * s.theta_dot * sin_t) / total_mass;
    const double theta_acc = (gravity * sin_t - cos_t * temp) /
                             (half_length * (4.0 / 3.0 - pole_mass * cos_t * cos_t / total_mass));
    const double x_acc = temp - pole_mass * half_length * theta_acc * cos_t / total_mass;
    s.x += dt * s.x_dot;
    s.x_dot += dt * x_acc;
    s.theta += dt * s.theta_dot;
    s.theta_dot += dt * theta_acc;
}

bool CartPole::failed() const { return std::abs(s.x) > x_limit || std::abs(s.theta) > theta_limit; }

std::vector<double> CartPole::observation() const {
    return {s.x / x_limit, s.x_dot / 3.0, s.theta / theta_limit, s.theta_dot / 3.0};
}

int cartpole_episode(const Genome& g, const Config& cfg, Rng& rng, int max_steps, double start_range) {
    auto net = Network::from_genome(g, cfg);
    CartPole cp;
    // Start state: x, x_dot, theta_dot ~ U(-r, r), theta ~ U(-r/2, r/2) (radians).
    cp.s = {rng.uniform(-start_range, start_range), rng.uniform(-start_range, start_range),
            rng.uniform(-start_range, start_range) * 0.5, rng.uniform(-start_range, start_range)};
    for (int t = 0; t < max_steps; ++t) {
        cp.step(net.activate(cp.observation())[0] > 0.5);
        if (cp.failed()) return t;
    }
    return max_steps;
}

// ---------------------------------------------------------------------------------------------- Double pole

namespace {
constexpr double kGravity = -9.8, kMup = 0.000002, kMassCart = 1.0, kMassPole1 = 0.1, kMassPole2 = 0.01;
constexpr double kLength1 = 0.5, kLength2 = 0.05, kTau = 0.01, kFailAngle = 0.6283185307179586;  // 36 deg

using Vec6 = std::array<double, 6>;

Vec6 derivative(const Vec6& y, double force) {
    const double x_dot = y[1], t1 = y[2], t1_dot = y[3], t2 = y[4], t2_dot = y[5];
    const double ml1 = kLength1 * kMassPole1, ml2 = kLength2 * kMassPole2;
    const double c1 = std::cos(t1), s1 = std::sin(t1), c2 = std::cos(t2), s2 = std::sin(t2);
    const double g1 = kGravity * s1, g2 = kGravity * s2;
    const double tmp1 = kMup * t1_dot / ml1, tmp2 = kMup * t2_dot / ml2;
    const double f1 = ml1 * t1_dot * t1_dot * s1 + 0.75 * kMassPole1 * c1 * (tmp1 + g1);
    const double f2 = ml2 * t2_dot * t2_dot * s2 + 0.75 * kMassPole2 * c2 * (tmp2 + g2);
    const double m1 = kMassPole1 * (1.0 - 0.75 * c1 * c1);
    const double m2 = kMassPole2 * (1.0 - 0.75 * c2 * c2);
    const double x_dd = (force + f1 + f2) / (m1 + m2 + kMassCart);
    const double t1_dd = -0.75 * (x_dd * c1 + g1 + tmp1) / kLength1;
    const double t2_dd = -0.75 * (x_dd * c2 + g2 + tmp2) / kLength2;
    return {x_dot, x_dd, t1_dot, t1_dd, t2_dot, t2_dd};
}
}  // namespace

void DoublePole::step(double force) {
    Vec6 y = {s.x, s.x_dot, s.theta1, s.theta1_dot, s.theta2, s.theta2_dot};
    for (int sub = 0; sub < 2; ++sub) {  // RK4, two integration steps per decision
        const Vec6 k1 = derivative(y, force);
        Vec6 tmp;
        for (int i = 0; i < 6; ++i) tmp[i] = y[i] + 0.5 * kTau * k1[i];
        const Vec6 k2 = derivative(tmp, force);
        for (int i = 0; i < 6; ++i) tmp[i] = y[i] + 0.5 * kTau * k2[i];
        const Vec6 k3 = derivative(tmp, force);
        for (int i = 0; i < 6; ++i) tmp[i] = y[i] + kTau * k3[i];
        const Vec6 k4 = derivative(tmp, force);
        for (int i = 0; i < 6; ++i) y[i] += kTau / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
    s = {y[0], y[1], y[2], y[3], y[4], y[5]};
}

bool DoublePole::failed() const {
    return std::abs(s.x) > 2.4 || std::abs(s.theta1) > kFailAngle || std::abs(s.theta2) > kFailAngle;
}

std::vector<double> DoublePole::observation() const {
    return {s.x / 2.4, s.x_dot / 1.5, s.theta1 / kFailAngle, s.theta1_dot / 2.0, s.theta2 / kFailAngle, s.theta2_dot / 2.0};
}

int double_pole_episode(const Genome& g, const Config& cfg, const DoublePoleState& start, int max_steps) {
    auto net = Network::from_genome(g, cfg);
    DoublePole dp;
    dp.s = start;
    for (int t = 0; t < max_steps; ++t) {
        const double out = net.activate(dp.observation())[0];
        dp.step((out - 0.5) * 20.0);  // sigmoid output -> force in [-10, 10] N
        if (dp.failed()) return t;
    }
    return max_steps;
}

// --------------------------------------------------------------------------------------------------- Dodge

DodgeGame::DodgeGame(std::uint64_t seed) : rng_(seed) {}

double DodgeGame::spawn_interval() const { return std::max(0.18, 0.5 - 0.01 * time); }

std::vector<double> DodgeGame::observation() const {
    std::vector<double> obs(2 * rows, 0.0);
    for (const auto& p : projectiles) {
        if (p.x > player_x + hit_distance) continue;  // already passed
        const double seconds_to_impact = std::max(0.0, (player_x - p.x) / p.speed);
        const double proximity = std::max(0.0, 1.0 - seconds_to_impact / 2.0);
        obs[p.row] = std::max(obs[p.row], proximity);
    }
    obs[rows + player_row] = 1.0;
    return obs;
}

bool DodgeGame::step(int action) {
    if (hit || time >= max_seconds) return false;
    cooldown_ = std::max(0.0, cooldown_ - dt);
    if (cooldown_ == 0.0 && action != 1) {
        const int target = player_row + (action == 0 ? -1 : 1);
        if (target >= 0 && target < rows) {
            player_row = target;
            cooldown_ = move_cooldown;
        }
    }
    spawn_timer_ += dt;
    if (spawn_timer_ >= spawn_interval()) {
        spawn_timer_ = 0.0;
        projectiles.push_back({10.0, rng_.index(rows), rng_.uniform(min_speed, max_speed)});
    }
    for (auto& p : projectiles) p.x += p.speed * dt;
    projectiles.erase(std::remove_if(projectiles.begin(), projectiles.end(), [](const Projectile& p) { return p.x > width + 10; }),
                      projectiles.end());
    for (const auto& p : projectiles)
        if (p.row == player_row && std::abs(p.x - player_x) < hit_distance) hit = true;
    time += dt;
    return !hit && time < max_seconds;
}

DodgeFrame DodgeGame::frame() const {
    DodgeFrame f{player_row, {}};
    for (const auto& p : projectiles) f.projectiles.emplace_back(p.x, p.row);
    return f;
}

double dodge_episode(const Genome& g, const Config& cfg, std::uint64_t seed, std::vector<DodgeFrame>* trace) {
    auto net = Network::from_genome(g, cfg);
    DodgeGame game(seed);
    do {
        if (trace) trace->push_back(game.frame());
        const auto out = net.activate(game.observation());
        const int action = static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin());
        if (!game.step(action)) break;
    } while (true);
    if (trace) trace->push_back(game.frame());
    return game.time;
}

// -------------------------------------------------------------------------------------------- Registry

const Environment& get(const std::string& name) {
    static const std::vector<Environment> all = [] {
        std::vector<Environment> v;

        Environment x;
        x.name = "xor";
        x.description = "XOR, the benchmark of the original NEAT paper";
        x.num_inputs = 2;
        x.num_outputs = 1;
        x.fitness_threshold = 15.9;
        x.fitness = [](const Genome& g, const Config& cfg, Rng&) { return xor_fitness(g, cfg); };
        x.solved = [](const Genome& g, const Config& cfg) { return xor_solved(g, cfg); };
        x.generalization = [](const Genome& g, const Config& cfg) { return xor_solved(g, cfg) ? 1.0 : 0.0; };
        v.push_back(x);

        Environment c;
        c.name = "cartpole";
        c.description = "single pole balancing (Barto et al. 1983), 2,000 steps = 40 s";
        c.num_inputs = 4;
        c.num_outputs = 1;
        c.fitness_threshold = 2000.0;
        c.fitness = [](const Genome& g, const Config& cfg, Rng& rng) {
            double total = 0.0;
            for (int e = 0; e < 3; ++e) total += cartpole_episode(g, cfg, rng, 2000, 0.05);
            return total / 3.0;
        };
        c.generalization = [](const Genome& g, const Config& cfg) {
            Rng rng(12345);  // unseen starts, wider than in training
            int ok = 0;
            for (int e = 0; e < 100; ++e) ok += cartpole_episode(g, cfg, rng, 2000, 0.2) == 2000;
            return ok / 100.0;
        };
        v.push_back(c);

        Environment dp;
        dp.name = "double_pole";
        dp.description = "double-pole balancing with velocities, the NEAT paper's control benchmark (solved = 100k steps)";
        dp.num_inputs = 6;
        dp.num_outputs = 1;
        dp.fitness_threshold = 100000.0;
        dp.fitness = [](const Genome& g, const Config& cfg, Rng&) {
            return static_cast<double>(double_pole_episode(g, cfg, DoublePoleState{}, 100000));
        };
        dp.generalization = [](const Genome& g, const Config& cfg) {
            // Fraction of 100 random starts balanced for 10,000 steps.
            Rng rng(2024);
            int ok = 0;
            for (int e = 0; e < 100; ++e) {
                DoublePoleState start;
                start.x = rng.uniform(-0.5, 0.5);
                start.x_dot = rng.uniform(-0.2, 0.2);
                start.theta1 = rng.uniform(-0.1, 0.1);
                start.theta1_dot = rng.uniform(-0.2, 0.2);
                ok += double_pole_episode(g, cfg, start, 10000) == 10000;
            }
            return ok / 100.0;
        };
        v.push_back(dp);

        Environment d;
        d.name = "dodge";
        d.description = "the 2017 dodge game: survive projectiles across 6 rows (max 60 s)";
        d.num_inputs = 2 * DodgeGame::rows;
        d.num_outputs = 3;
        d.fitness_threshold = DodgeGame::max_seconds;
        d.fitness = [](const Genome& g, const Config& cfg, Rng& rng) {
            double total = 0.0;
            for (int e = 0; e < 5; ++e) total += dodge_episode(g, cfg, rng.next_u64());
            return total / 5.0;
        };
        d.validation = [](const Genome& g, const Config& cfg) {
            double total = 0.0;
            for (int e = 0; e < 20; ++e) total += dodge_episode(g, cfg, 2000000 + e);
            return total / 20.0;
        };
        d.generalization = [](const Genome& g, const Config& cfg) {
            double total = 0.0;
            for (int e = 0; e < 50; ++e) total += dodge_episode(g, cfg, 1000000 + e);
            return total / 50.0 / DodgeGame::max_seconds;
        };
        d.defaults = {{"add_connection_prob", "0.3"}, {"add_node_prob", "0.05"}, {"population_size", "300"}, {"target_species", "8"}};
        v.push_back(d);
        return v;
    }();
    for (const auto& e : all)
        if (e.name == name) return e;
    throw std::invalid_argument("unknown environment: " + name);
}

std::vector<std::string> names() { return {"xor", "cartpole", "double_pole", "dodge"}; }

}  // namespace neat::envs

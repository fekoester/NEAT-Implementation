#pragma once

#include <cmath>
#include <cstdint>

namespace neat {

// Small, fast, portable RNG (xoshiro256**). Unlike std:: distributions, results are identical across
// compilers and platforms, so a seed reproduces a run everywhere.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 1) { reseed(seed); }

    void reseed(std::uint64_t seed) {
        std::uint64_t x = seed;
        for (auto& s : state_) s = splitmix64(x);
        has_spare_ = false;
    }

    std::uint64_t next_u64() {
        const std::uint64_t result = rotl(state_[1] * 5, 7) * 9;
        const std::uint64_t t = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= t;
        state_[3] = rotl(state_[3], 45);
        return result;
    }

    // Uniform in [0, 1).
    double uniform() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }
    double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
    bool chance(double p) { return uniform() < p; }

    // Uniform integer in [0, n). n must be > 0.
    std::uint64_t below(std::uint64_t n) {
        // Modulo bias is < n / 2^64: negligible for population-sized n. Portable (no __int128).
        return next_u64() % n;
    }
    int index(std::size_t n) { return static_cast<int>(below(n)); }

    // Standard normal (Box-Muller).
    double normal() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u = 0.0;
        while (u <= 1e-300) u = uniform();
        const double v = uniform();
        const double r = std::sqrt(-2.0 * std::log(u));
        constexpr double two_pi = 6.283185307179586476925286766559;
        spare_ = r * std::sin(two_pi * v);
        has_spare_ = true;
        return r * std::cos(two_pi * v);
    }
    double normal(double mean, double sigma) { return mean + sigma * normal(); }

    // Derives an independent stream, e.g. one per genome evaluation.
    static std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
        std::uint64_t x = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        return splitmix64(x);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    static std::uint64_t splitmix64(std::uint64_t& x) {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::uint64_t state_[4]{};
    double spare_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace neat

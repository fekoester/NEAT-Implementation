#include "neat/population.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace neat {

Population::Population(const Config& cfg)
    : cfg_(cfg), rng_(cfg.seed), tracker_(Genome::first_hidden_id(cfg)) {
    if (cfg.population_size < 2) throw std::invalid_argument("population_size must be >= 2");
    genomes_.reserve(cfg.population_size);
    for (int i = 0; i < cfg.population_size; ++i) genomes_.push_back(Genome::minimal(cfg_, tracker_, rng_));
}

void Population::evaluate_all(const Evaluator& evaluate) {
    const int n = static_cast<int>(genomes_.size());
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int threads = std::min(n, cfg_.threads > 0 ? cfg_.threads : static_cast<int>(hw));
    std::atomic<int> next{0};
    auto worker = [&] {
        for (int i = next++; i < n; i = next++) {
            Rng rng(Rng::mix(cfg_.seed, static_cast<std::uint64_t>(generation_) * 1000003ULL + i));
            genomes_[i].fitness = evaluate(genomes_[i], rng);
        }
    };
    if (threads <= 1) {
        worker();
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }
    evaluated_ = std::move(genomes_);
    genomes_.clear();
}

void Population::speciate() {
    for (auto& s : species_) s.members.clear();

    for (int i = 0; i < static_cast<int>(evaluated_.size()); ++i) {
        Genome& g = evaluated_[i];
        Species* home = nullptr;
        for (auto& s : species_) {
            if (Genome::compatibility(g, s.representative, cfg_) < cfg_.compatibility_threshold) {
                home = &s;
                break;
            }
        }
        if (!home) {
            Species s;
            s.id = next_species_id_++;
            s.representative = g;
            s.created = generation_;
            s.last_improved = generation_;
            species_.push_back(std::move(s));
            home = &species_.back();
        }
        home->members.push_back(i);
        g.species = home->id;
    }

    species_.erase(std::remove_if(species_.begin(), species_.end(), [](const Species& s) { return s.members.empty(); }),
                   species_.end());

    // Next generation is compared against a random member of each species (Stanley & Miikkulainen 2002).
    for (auto& s : species_) s.representative = evaluated_[s.members[rng_.index(s.members.size())]];

    if (cfg_.target_species > 0) {
        const int count = static_cast<int>(species_.size());
        if (count < cfg_.target_species)
            cfg_.compatibility_threshold = std::max(cfg_.threshold_min, cfg_.compatibility_threshold - cfg_.threshold_step);
        else if (count > cfg_.target_species)
            cfg_.compatibility_threshold += cfg_.threshold_step;
    }
}

std::vector<int> Population::allocate_offspring(const std::vector<double>& shares, int total) {
    // Largest-remainder rounding: the counts always sum to `total` (the 2017 code rounded each species
    // independently, so the population size drifted).
    std::vector<int> counts(shares.size(), 0);
    if (shares.empty()) return counts;
    double sum = std::accumulate(shares.begin(), shares.end(), 0.0);
    std::vector<double> exact(shares.size());
    for (std::size_t i = 0; i < shares.size(); ++i)
        exact[i] = sum > 0.0 ? shares[i] / sum * total : static_cast<double>(total) / shares.size();
    int assigned = 0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        counts[i] = static_cast<int>(std::floor(exact[i]));
        assigned += counts[i];
    }
    std::vector<std::size_t> order(shares.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return exact[a] - std::floor(exact[a]) > exact[b] - std::floor(exact[b]);
    });
    for (std::size_t k = 0; assigned < total; k = (k + 1) % order.size(), ++assigned) ++counts[order[k]];
    return counts;
}

std::vector<Genome> Population::reproduce() {
    tracker_.new_generation();

    // Stagnation bookkeeping.
    for (auto& s : species_) {
        double best = -1e300;
        for (int m : s.members) best = std::max(best, evaluated_[m].fitness);
        if (best > s.best_fitness) {
            s.best_fitness = best;
            s.last_improved = generation_;
        }
    }
    std::vector<int> by_best(species_.size());
    std::iota(by_best.begin(), by_best.end(), 0);
    std::sort(by_best.begin(), by_best.end(), [&](int a, int b) { return species_[a].best_fitness > species_[b].best_fitness; });
    std::vector<bool> alive(species_.size(), true);
    for (std::size_t rank = 0; rank < by_best.size(); ++rank) {
        const Species& s = species_[by_best[rank]];
        const bool protected_species = static_cast<int>(rank) < cfg_.protected_top_species;
        if (!protected_species && generation_ - s.last_improved >= cfg_.stagnation_generations) alive[by_best[rank]] = false;
    }

    // Explicit fitness sharing: adjusted = fitness / species size (fitness shifted to be non-negative).
    double min_fitness = 0.0;
    for (const auto& g : evaluated_) min_fitness = std::min(min_fitness, g.fitness);
    std::vector<double> shares(species_.size(), 0.0);
    for (std::size_t s = 0; s < species_.size(); ++s) {
        if (!alive[s]) continue;
        for (int m : species_[s].members) {
            Genome& g = evaluated_[m];
            g.adjusted_fitness = (g.fitness - min_fitness) / species_[s].members.size();
            shares[s] += g.adjusted_fitness;
        }
        if (shares[s] <= 0.0) shares[s] = 1e-12;  // alive species with zero fitness still get a chance
    }
    if (std::none_of(alive.begin(), alive.end(), [](bool a) { return a; })) {
        // Everything stagnated and nothing is protected: restart allocation over all species.
        for (std::size_t s = 0; s < species_.size(); ++s) shares[s] = 1.0;
    }
    const std::vector<int> offspring = allocate_offspring(shares, cfg_.population_size);

    // Parent pools: the top `survival_threshold` of each species.
    std::vector<std::vector<int>> pools(species_.size());
    for (std::size_t s = 0; s < species_.size(); ++s) {
        auto members = species_[s].members;
        std::sort(members.begin(), members.end(), [&](int a, int b) { return evaluated_[a].fitness > evaluated_[b].fitness; });
        const int keep = std::max(std::min<int>(2, members.size()),
                                  static_cast<int>(std::ceil(cfg_.survival_threshold * members.size())));
        pools[s].assign(members.begin(), members.begin() + keep);
    }

    std::vector<Genome> next;
    next.reserve(cfg_.population_size);
    for (std::size_t s = 0; s < species_.size(); ++s) {
        int remaining = offspring[s];
        if (remaining == 0) continue;
        const auto& pool = pools[s];

        // Elitism: the champion of a sufficiently large species is copied unchanged.
        if (static_cast<int>(species_[s].members.size()) >= cfg_.elitism_min_species_size) {
            next.push_back(evaluated_[pool.front()]);
            --remaining;
        }

        for (; remaining > 0; --remaining) {
            const Genome& mom = evaluated_[pool[rng_.index(pool.size())]];
            Genome child;
            if (rng_.chance(cfg_.crossover_prob)) {
                const Genome* dad = nullptr;
                if (species_.size() > 1 && rng_.chance(cfg_.interspecies_mating_prob)) {
                    std::size_t other = rng_.index(species_.size());
                    if (other == s) other = (other + 1) % species_.size();
                    dad = &evaluated_[pools[other][rng_.index(pools[other].size())]];
                } else {
                    dad = &evaluated_[pool[rng_.index(pool.size())]];
                }
                bool mom_fitter = mom.fitness > dad->fitness || (mom.fitness == dad->fitness && rng_.chance(0.5));
                child = mom_fitter ? Genome::crossover(mom, *dad, cfg_, rng_) : Genome::crossover(*dad, mom, cfg_, rng_);
            } else {
                child = mom;
            }
            child.mutate(cfg_, tracker_, rng_);
            child.fitness = child.adjusted_fitness = 0.0;
            child.species = -1;
            next.push_back(std::move(child));
        }
    }

    // Drop species that received no offspring (their representatives would otherwise linger).
    std::vector<Species> kept;
    for (std::size_t s = 0; s < species_.size(); ++s)
        if (offspring[s] > 0) kept.push_back(species_[s]);
    species_ = std::move(kept);
    return next;
}

GenerationStats Population::step(const Evaluator& evaluate) {
    evaluate_all(evaluate);
    speciate();

    GenerationStats stats;
    stats.generation = generation_;
    stats.num_species = static_cast<int>(species_.size());
    stats.compatibility_threshold = cfg_.compatibility_threshold;
    const Genome* best = &evaluated_.front();
    double sum = 0.0, hidden = 0.0, conns = 0.0;
    for (const auto& g : evaluated_) {
        sum += g.fitness;
        hidden += g.num_hidden();
        conns += g.num_enabled();
        if (g.fitness > best->fitness) best = &g;
    }
    const double n = static_cast<double>(evaluated_.size());
    stats.best_fitness = best->fitness;
    stats.mean_fitness = sum / n;
    stats.mean_hidden = hidden / n;
    stats.mean_connections = conns / n;
    stats.best_hidden = best->num_hidden();
    stats.best_connections = best->num_enabled();
    if (!best_ever_ || best->fitness > best_ever_->fitness) best_ever_ = *best;

    genomes_ = reproduce();
    ++generation_;
    return stats;
}

std::vector<GenerationStats> Population::run(const Evaluator& evaluate, int generations,
                                             const std::function<bool(const GenerationStats&)>& stop,
                                             const std::function<void(const GenerationStats&, const Population&)>& on_generation) {
    std::vector<GenerationStats> history;
    for (int g = 0; g < generations; ++g) {
        history.push_back(step(evaluate));
        if (on_generation) on_generation(history.back(), *this);
        if (stop && stop(history.back())) break;
    }
    return history;
}

}  // namespace neat

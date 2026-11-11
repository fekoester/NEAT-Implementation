// Train NEAT on a built-in environment.
//
//   neat_train --env xor --generations 100 --seed 1 --out runs/xor
//   neat_train --env dodge --generations 300 --set population_size=200 --out runs/dodge
//   neat_train --env xor --runs 100 --out runs/xor_benchmark      # repeated runs, summary statistics
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "../envs/envs.hpp"
#include "neat/io.hpp"
#include "neat/population.hpp"

namespace fs = std::filesystem;

namespace {

struct Args {
    std::string env = "xor";
    int generations = 100;
    std::uint64_t seed = 1;
    int runs = 1;
    std::string out = "runs/out";
    std::string config_file;
    std::vector<std::pair<std::string, std::string>> overrides;
    bool quiet = false;
};

[[noreturn]] void usage(int code) {
    std::cout << "usage: neat_train [--env xor|cartpole|double_pole|dodge] [--generations N] [--seed S] [--runs K]\n"
                 "                  [--config FILE] [--set key=value ...] [--out DIR] [--quiet]\n";
    std::exit(code);
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) usage(2);
            return argv[++i];
        };
        if (k == "--env") a.env = value();
        else if (k == "--generations") a.generations = std::stoi(value());
        else if (k == "--seed") a.seed = std::stoull(value());
        else if (k == "--runs") a.runs = std::stoi(value());
        else if (k == "--out") a.out = value();
        else if (k == "--config") a.config_file = value();
        else if (k == "--set") {
            const std::string kv = value();
            const auto eq = kv.find('=');
            if (eq == std::string::npos) usage(2);
            a.overrides.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else if (k == "--quiet") a.quiet = true;
        else if (k == "--help" || k == "-h") usage(0);
        else usage(2);
    }
    return a;
}

struct RunResult {
    int generations = 0;  // generations evaluated
    int solved_at = -1;   // generation index at which the threshold was reached
    double best_fitness = 0.0;
    double generalization = 0.0;
    int hidden = 0, connections = 0;
    long evaluations = 0;
    double seconds = 0.0;
};

RunResult run_once(const neat::envs::Environment& env, neat::Config cfg, const Args& args, const fs::path& dir, bool verbose) {
    fs::create_directories(dir);
    std::ofstream(dir / "config.txt") << cfg.to_string();
    std::ofstream csv(dir / "stats.csv");
    csv << "generation,best_fitness,mean_fitness,species,mean_hidden,mean_connections,best_hidden,best_connections,threshold\n";

    neat::Population pop(cfg);
    const auto start = std::chrono::steady_clock::now();
    RunResult result;
    auto evaluator = [&](const neat::Genome& g, neat::Rng& rng) { return env.fitness(g, cfg, rng); };
    bool solved = false;
    std::optional<neat::Genome> validated_champion;
    double best_validation = -1e300;
    pop.run(
        evaluator, args.generations,
        [&](const neat::GenerationStats&) { return solved; },
        [&](const neat::GenerationStats& s, const neat::Population& p) {
            const auto& gs = p.genomes();
            const auto best = std::max_element(gs.begin(), gs.end(), [](const auto& a, const auto& b) { return a.fitness < b.fitness; });
            if (env.validation) {
                const double v = env.validation(*best, cfg);
                if (v > best_validation) {
                    best_validation = v;
                    validated_champion = *best;
                }
            }
            if (env.solved) {
                solved = env.solved(*best, cfg);
            } else {
                solved = s.best_fitness >= env.fitness_threshold;
            }
            csv << s.generation << "," << s.best_fitness << "," << s.mean_fitness << "," << s.num_species << ","
                << s.mean_hidden << "," << s.mean_connections << "," << s.best_hidden << "," << s.best_connections << ","
                << s.compatibility_threshold << "\n";
            if (verbose)
                std::cout << "gen " << std::setw(4) << s.generation << "  best " << std::setw(10) << s.best_fitness
                          << "  mean " << std::setw(10) << s.mean_fitness << "  species " << std::setw(3) << s.num_species
                          << "  hidden " << s.best_hidden << "  conns " << s.best_connections << "\n";
            result.generations = s.generation + 1;
            if (result.solved_at < 0 && solved) result.solved_at = s.generation;
        });
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Champion: best validation score if the task has a validation set, otherwise best training fitness.
    const neat::Genome& best = validated_champion ? *validated_champion : pop.best_ever();
    result.best_fitness = best.fitness;
    result.generalization = env.generalization ? env.generalization(best, cfg) : 0.0;
    result.hidden = best.num_hidden();
    result.connections = best.num_enabled();
    result.evaluations = static_cast<long>(result.generations) * cfg.population_size;
    neat::save_genome(best, (dir / "champion.genome").string());
    std::ofstream dot(dir / "champion.dot");
    std::ostringstream title;
    title << env.name << " champion (fitness " << std::setprecision(4) << best.fitness << ")";
    neat::write_dot(best, dot, title.str());
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    const auto& env = neat::envs::get(args.env);

    neat::Config cfg = args.config_file.empty() ? neat::Config{} : neat::Config::from_file(args.config_file);
    cfg.num_inputs = env.num_inputs;
    cfg.num_outputs = env.num_outputs;
    if (args.config_file.empty())
        for (const auto& [k, v] : env.defaults) cfg.set(k, v);
    for (const auto& [k, v] : args.overrides) cfg.set(k, v);

    std::cout << "environment: " << env.name << " - " << env.description << "\n";
    std::vector<RunResult> results;
    std::ofstream runs_csv;
    if (args.runs > 1) {
        fs::create_directories(args.out);
        runs_csv.open(fs::path(args.out) / "runs.csv");
        runs_csv << "run,seed,solved_at,generations,evaluations,best_fitness,generalization,hidden,connections,seconds\n";
    }
    for (int r = 0; r < args.runs; ++r) {
        cfg.seed = args.seed + r;
        const fs::path dir = args.runs > 1 ? fs::path(args.out) / ("run_" + std::to_string(r)) : fs::path(args.out);
        const RunResult res = run_once(env, cfg, args, dir, !args.quiet && args.runs == 1);
        results.push_back(res);
        if (args.runs > 1) {
            runs_csv << r << "," << cfg.seed << "," << res.solved_at << "," << res.generations << "," << res.evaluations << ","
                     << res.best_fitness << "," << res.generalization << "," << res.hidden << "," << res.connections << ","
                     << res.seconds << "\n";
            if (!args.quiet)
                std::cout << "run " << r << ": " << (res.solved_at >= 0 ? "solved at gen " + std::to_string(res.solved_at) : "not solved")
                          << ", hidden " << res.hidden << ", " << std::fixed << std::setprecision(2) << res.seconds << " s\n";
        } else {
            std::cout << "best fitness " << res.best_fitness << ", generalization " << res.generalization << ", hidden nodes "
                      << res.hidden << ", connections " << res.connections << ", " << res.seconds << " s\n"
                      << "saved " << (dir / "champion.genome").string() << ", stats.csv, champion.dot\n";
        }
    }

    if (args.runs > 1) {
        std::vector<double> gens, evals, hidden;
        for (const auto& r : results)
            if (r.solved_at >= 0) {
                gens.push_back(r.solved_at + 1);
                evals.push_back(static_cast<double>(r.evaluations));
                hidden.push_back(r.hidden);
            }
        auto mean = [](const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
        auto sd = [&](const std::vector<double>& v) {
            if (v.size() < 2) return 0.0;
            const double m = mean(v);
            double s = 0.0;
            for (double x : v) s += (x - m) * (x - m);
            return std::sqrt(s / (v.size() - 1));
        };
        std::cout << std::fixed << std::setprecision(2) << "\nsolved " << gens.size() << "/" << results.size() << " runs; "
                  << "generations " << mean(gens) << " ± " << sd(gens) << ", evaluations " << mean(evals) << " ± " << sd(evals)
                  << ", hidden nodes " << mean(hidden) << " ± " << sd(hidden) << "\n";
    }
    return 0;
}

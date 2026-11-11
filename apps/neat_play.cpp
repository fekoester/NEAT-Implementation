// Replay a saved champion and print its scores; optionally record a dodge episode for rendering.
//
//   neat_play --env dodge --genome runs/dodge/champion.genome --seed 7 --trace dodge_trace.csv
#include <fstream>
#include <iostream>
#include <string>

#include "../envs/envs.hpp"
#include "neat/io.hpp"

int main(int argc, char** argv) {
    std::string env_name = "dodge", genome_path, trace_path;
    std::uint64_t seed = 7;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--env") env_name = v;
        else if (k == "--genome") genome_path = v;
        else if (k == "--seed") seed = std::stoull(v);
        else if (k == "--trace") trace_path = v;
    }
    if (genome_path.empty()) {
        std::cerr << "usage: neat_play --env NAME --genome FILE [--seed S] [--trace FILE.csv]\n";
        return 2;
    }
    const auto& env = neat::envs::get(env_name);
    neat::Config cfg;
    cfg.num_inputs = env.num_inputs;
    cfg.num_outputs = env.num_outputs;
    const neat::Genome g = neat::load_genome(genome_path);

    if (env_name == "dodge") {
        std::vector<neat::envs::DodgeFrame> trace;
        const double seconds = neat::envs::dodge_episode(g, cfg, seed, trace_path.empty() ? nullptr : &trace);
        std::cout << "survived " << seconds << " s (seed " << seed << ")\n";
        if (!trace_path.empty()) {
            std::ofstream out(trace_path);
            out << "frame,player_row,projectile_x,projectile_row\n";
            for (std::size_t f = 0; f < trace.size(); ++f) {
                out << f << "," << trace[f].player_row << ",,\n";
                for (const auto& [x, row] : trace[f].projectiles) out << f << "," << trace[f].player_row << "," << x << "," << row << "\n";
            }
            std::cout << "trace written to " << trace_path << "\n";
        }
    }
    if (env.generalization) std::cout << "generalization score: " << env.generalization(g, cfg) << "\n";
    return 0;
}

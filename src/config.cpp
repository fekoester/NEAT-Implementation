#include "neat/config.hpp"

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <stdexcept>

namespace neat {

std::string to_string(Activation a) {
    switch (a) {
        case Activation::Sigmoid: return "sigmoid";
        case Activation::Tanh: return "tanh";
        case Activation::ReLU: return "relu";
        case Activation::Identity: return "identity";
    }
    return "sigmoid";
}

Activation activation_from_string(const std::string& s) {
    if (s == "sigmoid") return Activation::Sigmoid;
    if (s == "tanh") return Activation::Tanh;
    if (s == "relu") return Activation::ReLU;
    if (s == "identity") return Activation::Identity;
    throw std::invalid_argument("unknown activation: " + s);
}

namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

bool parse_bool(const std::string& v) {
    if (v == "true" || v == "1" || v == "yes") return true;
    if (v == "false" || v == "0" || v == "no") return false;
    throw std::invalid_argument("expected a boolean, got " + v);
}

// key -> (setter, getter)
using Field = std::pair<std::function<void(Config&, const std::string&)>, std::function<std::string(const Config&)>>;

std::map<std::string, Field>& fields() {
    static std::map<std::string, Field> f;
    if (!f.empty()) return f;
#define NEAT_INT(name) f[#name] = {[](Config& c, const std::string& v) { c.name = std::stoi(v); }, [](const Config& c) { return std::to_string(c.name); }}
#define NEAT_U64(name) f[#name] = {[](Config& c, const std::string& v) { c.name = std::stoull(v); }, [](const Config& c) { return std::to_string(c.name); }}
#define NEAT_DBL(name) f[#name] = {[](Config& c, const std::string& v) { c.name = std::stod(v); }, [](const Config& c) { std::ostringstream o; o << c.name; return o.str(); }}
#define NEAT_BOOL(name) f[#name] = {[](Config& c, const std::string& v) { c.name = parse_bool(v); }, [](const Config& c) { return std::string(c.name ? "true" : "false"); }}
#define NEAT_ACT(name) f[#name] = {[](Config& c, const std::string& v) { c.name = activation_from_string(v); }, [](const Config& c) { return to_string(c.name); }}
    NEAT_INT(num_inputs); NEAT_INT(num_outputs); NEAT_BOOL(feed_forward);
    NEAT_ACT(hidden_activation); NEAT_ACT(output_activation); NEAT_DBL(sigmoid_steepness);
    NEAT_INT(population_size); NEAT_U64(seed); NEAT_INT(threads);
    NEAT_DBL(init_weight_range); NEAT_DBL(weight_mutate_prob); NEAT_DBL(weight_perturb_prob);
    NEAT_DBL(weight_perturb_sigma); NEAT_DBL(weight_replace_range); NEAT_DBL(weight_limit);
    NEAT_DBL(add_node_prob); NEAT_DBL(add_connection_prob); NEAT_DBL(toggle_enable_prob); NEAT_INT(add_connection_attempts);
    NEAT_DBL(crossover_prob); NEAT_DBL(interspecies_mating_prob); NEAT_DBL(disabled_gene_inherit_prob);
    NEAT_DBL(survival_threshold); NEAT_INT(elitism_min_species_size);
    NEAT_DBL(c_excess); NEAT_DBL(c_disjoint); NEAT_DBL(c_weight); NEAT_DBL(compatibility_threshold);
    NEAT_INT(normalize_min_genes); NEAT_INT(target_species); NEAT_DBL(threshold_step); NEAT_DBL(threshold_min);
    NEAT_INT(stagnation_generations); NEAT_INT(protected_top_species);
#undef NEAT_INT
#undef NEAT_U64
#undef NEAT_DBL
#undef NEAT_BOOL
#undef NEAT_ACT
    return f;
}

}  // namespace

void Config::set(const std::string& key, const std::string& value) {
    auto it = fields().find(key);
    if (it == fields().end()) throw std::invalid_argument("unknown config key: " + key);
    it->second.first(*this, value);
}

std::string Config::to_string() const {
    std::ostringstream out;
    for (const auto& [key, field] : fields()) out << key << " = " << field.second(*this) << "\n";
    return out.str();
}

Config Config::from_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config file " + path);
    Config cfg;
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) throw std::invalid_argument(path + ":" + std::to_string(number) + ": expected key = value");
        cfg.set(trim(line.substr(0, eq)), trim(line.substr(eq + 1)));
    }
    return cfg;
}

}  // namespace neat

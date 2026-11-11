#include "neat/io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace neat {

namespace {

const char* type_name(NodeType t) {
    switch (t) {
        case NodeType::Input: return "input";
        case NodeType::Bias: return "bias";
        case NodeType::Output: return "output";
        case NodeType::Hidden: return "hidden";
    }
    return "hidden";
}

NodeType type_from(const std::string& s) {
    if (s == "input") return NodeType::Input;
    if (s == "bias") return NodeType::Bias;
    if (s == "output") return NodeType::Output;
    if (s == "hidden") return NodeType::Hidden;
    throw std::invalid_argument("unknown node type: " + s);
}

}  // namespace

void save_genome(const Genome& genome, std::ostream& out) {
    out.precision(17);
    out << "neat-genome 1\nfitness " << genome.fitness << "\n";
    for (const auto& n : genome.nodes) out << "node " << n.id << " " << type_name(n.type) << " " << to_string(n.activation) << "\n";
    for (const auto& c : genome.conns)
        out << "conn " << c.innovation << " " << c.in << " " << c.out << " " << c.weight << " " << (c.enabled ? 1 : 0) << "\n";
}

void save_genome(const Genome& genome, const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    save_genome(genome, out);
}

Genome load_genome(std::istream& in) {
    std::string header;
    int version = 0;
    in >> header >> version;
    if (header != "neat-genome" || version != 1) throw std::runtime_error("not a neat-genome v1 file");
    Genome g;
    std::string kind;
    while (in >> kind) {
        if (kind == "fitness") {
            in >> g.fitness;
        } else if (kind == "node") {
            NodeGene n;
            std::string type, act;
            in >> n.id >> type >> act;
            n.type = type_from(type);
            n.activation = activation_from_string(act);
            g.nodes.push_back(n);
        } else if (kind == "conn") {
            ConnectionGene c;
            int enabled = 1;
            in >> c.innovation >> c.in >> c.out >> c.weight >> enabled;
            c.enabled = enabled != 0;
            g.conns.push_back(c);
        } else {
            throw std::runtime_error("unexpected token in genome file: " + kind);
        }
    }
    g.sort_genes();
    return g;
}

Genome load_genome(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    return load_genome(in);
}

void write_dot(const Genome& genome, std::ostream& out, const std::string& title) {
    out << "digraph neat {\n  rankdir=LR;\n  splines=true;\n  node [shape=circle, style=filled, fontname=Helvetica, fontsize=10];\n";
    if (!title.empty()) out << "  labelloc=t; label=\"" << title << "\";\n";
    out << "  { rank=source; ";
    for (const auto& n : genome.nodes)
        if (n.type == NodeType::Input || n.type == NodeType::Bias) out << "n" << n.id << "; ";
    out << "}\n  { rank=sink; ";
    for (const auto& n : genome.nodes)
        if (n.type == NodeType::Output) out << "n" << n.id << "; ";
    out << "}\n";
    for (const auto& n : genome.nodes) {
        const char* color = n.type == NodeType::Input ? "#9ecae1" : n.type == NodeType::Bias ? "#d9d9d9"
                          : n.type == NodeType::Output ? "#fdae6b" : "#a1d99b";
        std::string label = n.type == NodeType::Bias ? "bias" : std::to_string(n.id);
        out << "  n" << n.id << " [label=\"" << label << "\", fillcolor=\"" << color << "\"];\n";
    }
    for (const auto& c : genome.conns) {
        const double width = 0.5 + std::min(4.0, std::abs(c.weight));
        out << "  n" << c.in << " -> n" << c.out << " [color=\"" << (c.weight >= 0 ? "#2b8cbe" : "#de2d26")
            << "\", penwidth=" << width << (c.enabled ? "" : ", style=dashed, color=\"#bdbdbd\"") << "];\n";
    }
    out << "}\n";
}

}  // namespace neat

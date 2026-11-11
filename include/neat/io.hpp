#pragma once

#include <iosfwd>
#include <string>

#include "neat/genome.hpp"

namespace neat {

// Plain-text genome format:
//   neat-genome 1
//   fitness <value>
//   node <id> <input|bias|output|hidden> <activation>
//   conn <innovation> <in> <out> <weight> <0|1>
void save_genome(const Genome& genome, std::ostream& out);
void save_genome(const Genome& genome, const std::string& path);
Genome load_genome(std::istream& in);
Genome load_genome(const std::string& path);

// Graphviz DOT drawing of the network (inputs left, outputs right; red = negative weight,
// line width ~ |weight|, disabled connections dashed). Render with: dot -Tpng net.dot -o net.png
void write_dot(const Genome& genome, std::ostream& out, const std::string& title = "");

}  // namespace neat

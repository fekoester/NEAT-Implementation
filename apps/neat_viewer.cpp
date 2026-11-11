// Live SFML viewer: a champion plays the dodge game while its network is drawn next to it.
// Built only when SFML is found (option NEAT_BUILD_VIEWER).
//
//   neat_viewer --genome runs/dodge/champion.genome [--seed 7] [--font legacy/CurlyLou.ttf]
//               [--screenshot out.png --frames 600]   (save a screenshot after N frames and exit)
//   Keys: R = restart with the next seed, Esc = quit.
#include <SFML/Graphics.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

#include "../envs/envs.hpp"
#include "neat/io.hpp"
#include "neat/network.hpp"

using neat::envs::DodgeGame;

namespace {

constexpr float kGameW = 720.f, kGameH = 480.f, kNetW = 420.f;

// Node positions: inputs left, outputs right, hidden nodes by depth in between.
std::map<int, sf::Vector2f> layout(const neat::Genome& g) {
    std::map<int, int> depth;
    for (const auto& n : g.nodes) depth[n.id] = (n.type == neat::NodeType::Input || n.type == neat::NodeType::Bias) ? 0 : 1;
    for (std::size_t iter = 0; iter < g.nodes.size(); ++iter)
        for (const auto& c : g.conns)
            if (c.enabled) depth[c.out] = std::max(depth[c.out], depth[c.in] + 1);
    int max_depth = 1;
    for (auto& [id, d] : depth) max_depth = std::max(max_depth, d);
    for (const auto& n : g.nodes)
        if (n.type == neat::NodeType::Output) depth[n.id] = max_depth;

    std::map<int, std::vector<int>> columns;
    for (const auto& n : g.nodes) columns[depth[n.id]].push_back(n.id);
    std::map<int, sf::Vector2f> pos;
    for (auto& [d, ids] : columns)
        for (std::size_t i = 0; i < ids.size(); ++i)
            pos[ids[i]] = {kGameW + 40.f + (kNetW - 80.f) * d / max_depth, 40.f + (kGameH - 80.f) * (i + 0.5f) / ids.size()};
    return pos;
}

}  // namespace

int main(int argc, char** argv) {
    std::string genome_path, font_path = "legacy/CurlyLou.ttf", screenshot;
    std::uint64_t seed = 7;
    int screenshot_frame = 600, frame = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i], v = argv[i + 1];
        if (k == "--genome") genome_path = v;
        else if (k == "--seed") seed = std::stoull(v);
        else if (k == "--font") font_path = v;
        else if (k == "--screenshot") screenshot = v;
        else if (k == "--frames") screenshot_frame = std::stoi(v);
    }
    if (genome_path.empty()) {
        std::cerr << "usage: neat_viewer --genome FILE [--seed S] [--font FILE.ttf]\n";
        return 2;
    }
    const auto& env = neat::envs::get("dodge");
    neat::Config cfg;
    cfg.num_inputs = env.num_inputs;
    cfg.num_outputs = env.num_outputs;
    const neat::Genome genome = neat::load_genome(genome_path);
    const auto positions = layout(genome);

    sf::RenderWindow window(sf::VideoMode(static_cast<unsigned>(kGameW + kNetW), static_cast<unsigned>(kGameH)), "NEAT dodge");
    window.setFramerateLimit(screenshot.empty() ? 60 : 0);
    sf::Font font;
    const bool has_font = font.loadFromFile(font_path);

    auto net = neat::Network::from_genome(genome, cfg);
    DodgeGame game(seed);
    const float row_h = kGameH / DodgeGame::rows;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape) window.close();
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::R) {
                game = DodgeGame(++seed);
                net.reset();
            }
        }
        const auto out = net.activate(game.observation());
        if (!game.hit && game.time < DodgeGame::max_seconds)
            game.step(static_cast<int>(std::max_element(out.begin(), out.end()) - out.begin()));

        window.clear(sf::Color(250, 250, 250));
        for (int r = 0; r < DodgeGame::rows; ++r) {
            sf::RectangleShape lane({kGameW, row_h - 2.f});
            lane.setPosition(0.f, r * row_h + 1.f);
            lane.setFillColor(r % 2 ? sf::Color(235, 238, 245) : sf::Color(244, 246, 250));
            window.draw(lane);
        }
        for (const auto& p : game.projectiles) {
            sf::CircleShape c(7.f);
            c.setOrigin(7.f, 7.f);
            c.setPosition(static_cast<float>(p.x), (p.row + 0.5f) * row_h);
            c.setFillColor(sf::Color(222, 45, 38));
            window.draw(c);
        }
        sf::CircleShape player(9.f);
        player.setOrigin(9.f, 9.f);
        player.setPosition(static_cast<float>(DodgeGame::player_x), (game.player_row + 0.5f) * row_h);
        player.setFillColor(game.hit ? sf::Color::Black : sf::Color(43, 140, 190));
        window.draw(player);

        for (const auto& c : genome.conns) {
            if (!c.enabled) continue;
            const sf::Color col = c.weight >= 0 ? sf::Color(43, 140, 190, 160) : sf::Color(222, 45, 38, 160);
            sf::Vertex line[] = {sf::Vertex(positions.at(c.in), col), sf::Vertex(positions.at(c.out), col)};
            window.draw(line, 2, sf::Lines);
        }
        for (const auto& n : genome.nodes) {
            sf::CircleShape c(8.f);
            c.setOrigin(8.f, 8.f);
            c.setPosition(positions.at(n.id));
            c.setFillColor(n.type == neat::NodeType::Output ? sf::Color(253, 174, 107)
                           : n.type == neat::NodeType::Hidden ? sf::Color(161, 217, 155) : sf::Color(158, 202, 225));
            window.draw(c);
        }
        if (has_font) {
            sf::Text text("t = " + std::to_string(static_cast<int>(game.time)) + " s   seed " + std::to_string(seed) +
                              (game.hit ? "   HIT - press R" : ""),
                          font, 18);
            text.setFillColor(sf::Color(40, 40, 40));
            text.setPosition(10.f, 6.f);
            window.draw(text);
        }
        if (!screenshot.empty() && ++frame == screenshot_frame) {
            sf::Texture texture;
            texture.create(window.getSize().x, window.getSize().y);
            texture.update(window);
            texture.copyToImage().saveToFile(screenshot);
            window.close();
        }
        window.display();
    }
    return 0;
}

#include "GameData.h"
#include <SFML/Graphics.hpp>
#include "Organism.h"

#pragma once

class GraphicHandler
{
public:
	GraphicHandler();
	~GraphicHandler();

	
	//Print an organism
	void printOrganism(
		Organism &organism);


	//Prints the game
	void printGame(
		vector<projectile> &projectiles,
		playerStruct &player);


	//Grabs the events on both windows
	void getEvents();

private:

	//The Font of the text
	sf::Font m_font;

	//The Windows for the neural network and the game
	sf::RenderWindow m_neuralNetworkWindow;
	sf::RenderWindow m_gameWindow;

};


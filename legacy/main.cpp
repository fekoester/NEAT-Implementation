#include <iostream>
#include "GraphicHandler.h"
#include "Generation.h"
#include "Game.h"

using namespace std;


//Processes the Graphical Interface of the neural network and the game
void processGraphicHandler(
	GraphicHandler &graphicHandler,
	Organism &organism,
	Game &game)
{
	graphicHandler.printOrganism(organism);
	graphicHandler.printGame(game.getProjectiles(), game.getPlayer());
	graphicHandler.getEvents();
}


//Processes the logic of the game and returns true if the player was hit by a projectile
//Also takes an organism and feeds its output to the game
bool processGame(
	Game &game,
	Organism &organism)
{
	return game.processLogic(organism.getOutputs());
}


//Processes the neural network of the organism and feed it fitness
void processOrganism(
	Organism &organism,
	Game &game)
{
	organism.addFitness(game.returnFitnessDelta());
	organism.feedInput(game.getInput());
	organism.processNeuralNetwork();
}


//MAIN
int main(int argc, char *argv[])
{
	Generation generation;
	GraphicHandler graphicHandler;
	Game game;
	int generationNumber = 0;
	int organismNumber = 0;
	int specieNumber = 0;
	double highestFitness = 0.0;
	int numberOrganisms = 0;

	/*
	generation.getOrganismByIndex(0, 0).printInfo();
	int c;
	cin >> c;
	generation.getOrganismByIndex(1, 0).printInfo();

	Organism testChild = generation.breedNewOrganism(generation.getOrganismByIndex(0, 0), generation.getOrganismByIndex(1, 0));
	cin >> c;
	testChild.printInfo();
	cin >> c;*/

	while (true)
	{
		for (vector<vector<Organism>>::iterator it = generation.getSpecies().begin(); it != generation.getSpecies().end(); ++it)
		{
			for (vector<Organism>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
			{
				numberOrganisms++;
			}
		}
		for (vector<vector<Organism>>::iterator it = generation.getSpecies().begin(); it != generation.getSpecies().end(); ++it)
		{
			for (vector<Organism>::iterator it2 = it->begin(); it2 != it->end(); ++it2)
			{
				while (!processGame(game, *it2))
				{
					processGraphicHandler(graphicHandler, *it2, game);
					if (game.printToTerminal()) { 					
						it2->printInfo(organismNumber);
						cout << "Highest Fitness so far: " << highestFitness << endl;
						cout << "Number of Species in this generation: " << generation.getSpecies().size() << endl;
						cout << "Number of Organisms in this generation: " << numberOrganisms << endl;
					}
					processOrganism(*it2, game);
				}
				game.cleanGame();
				organismNumber++;
				if (it2->getFitness() > highestFitness)
				{
					highestFitness = it2->getFitness();
				}
			}
			specieNumber++;
		}
		Generation newGeneration(generation);
		generation = newGeneration;
		generationNumber++;
		organismNumber = 0;
		specieNumber = 0;
		numberOrganisms = 0;
	}
	return 0;
}
#pragma once

#include <vector>
#include <time.h>
#include <random>
#include "GameData.h"

using namespace std;

class Game
{
public:
	Game();
	~Game();


	//Processes the velocities of the projectiles
	void processVelocities();


	//Process player moves
	void processPlayer(
		const vector<double> &outputs);


	//Processes the logic of the game
	bool processLogic(
		vector<double> outputs);


	//Collision checks
	bool collisionCheck();


	//Processes the time
	void processTime();


	//Creates a new projectile with a random velocity
	projectile createProjectile();


	//Destroy projectiles that are out of the map
	void destroyOutOfBoundaryProjectile();


	//Returns the player struct
	playerStruct &getPlayer();


	//Returns the projectiles
	vector<projectile> &getProjectiles();


	//Cleans the game from every projectile for a fresh new start
	void cleanGame();


	//Checks the time if Information should be printed to the terminal
	bool printToTerminal();


	//Returns a Fitness value that is the delta value since last time
	double returnFitnessDelta();


	//Gets the input of the game for the neural network
	vector<double> getInput();


	//Random generator 
	double randomReal(
		const double lowerBoundary,
		const double upperBoundary);

	int randomInt(
		const int lowerBoundary,
		const int upperBoundary);

	static random_device seed_generator;
	static unsigned seed;
	static mt19937 mersenne_generator;

private:
	playerStruct m_player;
	vector<projectile> m_projectiles;

	//Time
	clock_t m_deltaClock = 0;
	clock_t m_lastClock = 0;
	clock_t m_elapsedClockTerminalPrint = 0;
	clock_t m_elapsedClockProjectile = 0;
};


#include "Game.h"



Game::Game()
{
	//Initialize the player
	m_player.positionX = WIDTH_GAME - 3 * PLAYER_RADIUS;
	m_player.positionY = 2;
}


Game::~Game()
{
}


//Processes the velocities of the projectiles
void Game::processVelocities()
{
	for (vector<projectile>::iterator it = m_projectiles.begin(); it != m_projectiles.end(); ++it) {
		it->positionX += it->velocity * (float(m_deltaClock) / CLOCKS_PER_SEC);
	}
}


//Process player moves
void Game::processPlayer(
	const vector<double> &outputs)
{
	if (outputs[0] < 0.0)
	{
		m_player.positionY -= 1;
		if (m_player.positionY <= 0)
			m_player.positionY = 0;
	}
	if (outputs[0] > 0.0)
	{
		m_player.positionY += 1;
		if (m_player.positionY >= NUMBER_ROWS)
			m_player.positionY = NUMBER_ROWS;
	}
}


//Processes the logic of the game
bool Game::processLogic(
	vector<double> outputs)
{
	//Processes the time
	processTime();
	//Process projectiles...
	if ((float(m_elapsedClockProjectile) / CLOCKS_PER_SEC) > PROJECTILE_SPAWN_TIME) {
		projectile tmp = createProjectile();
		m_elapsedClockProjectile = 0;
		m_projectiles.push_back(tmp);
	}
	processVelocities();
	destroyOutOfBoundaryProjectile();
	bool gameLost = collisionCheck();
	//Adding fitness for the elapsed time
	processPlayer(outputs);

	return gameLost;
}


//Collision checks
bool Game::collisionCheck()
{
	for (vector<projectile>::iterator it = m_projectiles.begin(); it != m_projectiles.end(); ++it) {
		if (it->positionY == m_player.positionY && fabs(it->positionX - m_player.positionX) < PLAYER_RADIUS) {
			return true;
		}
	}
	return false;
}


//Processes the time
void Game::processTime()
{
	//Process time...
	m_deltaClock = clock() - m_lastClock;
	m_lastClock = clock();
	m_elapsedClockProjectile += m_deltaClock;
	m_elapsedClockTerminalPrint += m_deltaClock;
}


//Creates a new projectile with a random velocity
projectile Game::createProjectile()
{
	projectile tmp;
	tmp.positionY = randomInt(0.0, NUMBER_ROWS);
	tmp.velocity = randomReal(MIN_VELOCITY, MAX_VELOCITY);
	tmp.positionX = 10.0;

	return tmp;
}


//Destroy projectiles that are out of the map
void Game::destroyOutOfBoundaryProjectile()
{
	vector<projectile>::iterator it = m_projectiles.begin();
	while (it != m_projectiles.end()) 
	{
		if (it->positionX >= WIDTH_GAME + PROJECTILE_RADIUS) 
		{
			it = m_projectiles.erase(it);
		}
		else ++it;
	}
}


//Returns the player struct
playerStruct &Game::getPlayer()
{
	return m_player;
}


//Returns the projectiles
vector<projectile> &Game::getProjectiles()
{
	return m_projectiles;
}


//Cleans the game from every projectile for a fresh new start
void Game::cleanGame()
{
	m_projectiles.clear();
}


//Returns a Fitness value that is the delta value since last time
double Game::returnFitnessDelta()
{
	return (double(m_deltaClock) / CLOCKS_PER_SEC);
}


//Checks the time if Information should be printed to the terminal
bool Game::printToTerminal()
{
	if ((float(m_elapsedClockTerminalPrint) / CLOCKS_PER_SEC) > 2.5)
	{
		m_elapsedClockTerminalPrint = 0;
		return true;
	}
	return false;
}


//Gets the input of the game for the neural network
vector<double> Game::getInput()
{
	vector<vector<projectile>::iterator> frontProjectiles;
	for (vector<projectile>::iterator it = m_projectiles.begin(); it != m_projectiles.end(); ++it) {
		if (frontProjectiles.size() < (int)(NUMBER_OF_INDEPENDENT_VARAIABLES / 2.0)) {
			frontProjectiles.push_back(it);
		}
		else {
			double xValue = 0.0;
			vector<vector<projectile>::iterator>::iterator tmpIt;
			for (vector<vector<projectile>::iterator>::iterator it2 = frontProjectiles.begin(); it2 != frontProjectiles.end(); ++it2) {
				if ((**it2).positionX > xValue) {
					if ((*it).positionX < xValue) {
						(*it2) = it;
						break;
					}
					else {
						xValue = (**it2).positionX;
					}

				}
			}
		}
	}
	vector<double> inputVector;
	for (vector<vector<projectile>::iterator>::iterator it = frontProjectiles.begin(); it != frontProjectiles.end(); ++it) {
		double tmpX = 0.0;
		int tmpY = 0.0;
		tmpX = ((**it).positionX / 100) + (**it).velocity;
		tmpY = ((**it).positionY);
		inputVector.push_back(tmpX);
		inputVector.push_back(tmpY);
	}
	while (inputVector.size() < NUMBER_OF_INDEPENDENT_VARAIABLES - 1) {
		inputVector.push_back(0.0);
	}
	inputVector.push_back(m_player.positionY);
	return inputVector;
}


//Random engine call to generate a floating point number between lowerBoundary and upperBoundary
double Game::randomReal(
	const double lowerBoundary,
	const double upperBoundary)
{
	uniform_real_distribution<double> distribution_real(lowerBoundary, upperBoundary);
	return distribution_real(mersenne_generator);
}


//Random engine call to generate a integer between lowerBoundary and upperBoundary
int Game::randomInt(
	const int lowerBoundary,
	const int upperBoundary)
{
	uniform_int_distribution<int> distribution_int(lowerBoundary, upperBoundary);
	return distribution_int(mersenne_generator);
}


//Random engine initialisation
random_device Game::seed_generator;
unsigned Game::seed = seed_generator();
mt19937 Game::mersenne_generator(Game::seed);
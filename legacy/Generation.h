#pragma once

#include <vector>
#include <random>
#include <string>
#include "GameData.h"
#include "Organism.h"

using namespace std;

class Generation
{
public:
	//Creates the starting generation
	Generation();


	//Creates a new generation from the old one
	Generation(Generation &lastGeneration);


	//Creates a vector of organisms that represent the species of the last generation
	vector<Organism> createCompatibilitySpecies(
		Generation &lastGeneration);

	~Generation();


	//Breeds a new Organism by combining the Genomes of the father and mother and adds random mutations
	Organism breedNewOrganism(
		Organism &father,
		Organism &mother);


	//Compares  the compatibility of two Organism for the speciation of NEAT
	double compatibilityDistance(
		Organism &organism1,
		Organism &organism2);


	//Add a connection
	void addConnection(Genome &genome, int connFrom, int connTo);


	//Process all mutationchances
	void processMutation(Genome &genome);


	//Mutates an additional connection
	void mutateAddConnection(Genome &genome);


	//Mutates the weight of a connection
	void mutateConnection(Genome &genome);


	//Mutates a connection to enabled or disabled
	void mutateConnectionEnabling(ConnectionGene &connection);


	//Mutation for additional node
	void mutateNode(Genome &genome);


	//Mutation for a type change
	void mutateType(Genome &genome);


	//Checks the other species for same mutated connections to keep historical marking organized
	void Generation::checkHistoricalMarkings(
		ConnectionGene &newConnection);


	//Gets an organism out of this generation by index
	Organism &getOrganismByIndex(
		int indexSpecies,
		int indexOrganism);


	//Gets an organism out of this generation by index
	vector<vector<Organism>> &getSpecies();


	//Returns the historical Marking of this generation
	int getHistoricalMarking() { return m_historicalMarking; }


	//Returns the shared fitness of a n organism with its niche
	double getSharedFitness(
		int indexSpecies, 
		int indexOrganism);


	//Returns the sum of the shared fitness of a species
	double getSharedFitnessSpecies(
		int indexSpecies);


	//Returns the sum of all shared fitness of all species of this generation
	double getSumOfFitnessOfAllSpecies();


	//Returns a Species by Index
	vector<Organism> getSpeciesByIndex(
		int index);


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

	vector<vector<Organism>> m_species;


	int m_historicalMarking = 0;
};


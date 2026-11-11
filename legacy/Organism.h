#pragma once

#include "Genome.h"

class Organism
{
public:
	Organism(Genome &genome);
	~Organism();

	//Getter for the genome
	Genome &getGenome()
	{
		return m_genome;
	}


	//Feeds the input into the inputnodes
	void Organism::feedInput(
		vector<double> &inputs);


	//Processes the in- and ouputs of every node of the network
	void Organism::processNeuralNetwork();


	//Returns the output of the Organism as a vector of doubles
	vector<double> getOutputs();


	//Print this species information about their nodes and connections
	void printInfo(
		int organismNumber);


	//Returns the highest innovation number of this Species
	int getHighestInnovationNumber();


	//Threshold functions
	//Linear threshold
	double linear(
		const double &sum);


	//Sigmoid threshold
	double sigmoid(
		const double &sum);


	//Getter/Setter and Adder for fitness value
	void setFitness(
		double fitness)
	{
		m_fitness = fitness;
	}


	double getFitness()
	{
		return m_fitness;
	}


	void addFitness(
		double addValue)
	{
		m_fitness += addValue;
	}


private:
	//Genome of this organism
	Genome m_genome;

	//Data of the organism
	double m_fitness = 0.0;
};


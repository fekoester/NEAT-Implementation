#include <iostream>
#include "Organism.h"



Organism::Organism(Genome &genome):
	m_genome(genome)
{
	m_fitness = 0.0;
}


Organism::~Organism()
{
}


//Feeds the input into the inputnodes
void Organism::feedInput(
	vector<double> &inputs)
{
	int index = 0;
	if (inputs.size() != m_genome.m_inputNodes)
	{
		perror("Number of input-values is not equal to number of input nodes! Assuming all other inputs are 0...");
	}

	//Iterate through every node and push_back the inpts
	for (vector<NodeGene>::iterator it = m_genome.m_NodeGenes.begin(); it != m_genome.m_NodeGenes.end(); ++it)
	{
		if (it->nodeType == 0)
		{
			if (index >= inputs.size()) {
				it->inputs.push_back(0.0);
			}
			else {
				it->inputs.push_back(inputs[index]);
			}
			index++;
		}
	}
}


//Processes the in- and ouputs of every node of the network
void Organism::processNeuralNetwork() 
{
	for (vector<NodeGene>::iterator it = m_genome.m_NodeGenes.begin(); it != m_genome.m_NodeGenes.end(); ++it)
	{
		double sumInputs = 0.0;
		for (vector<double>::iterator it2 = it->inputs.begin(); it2 != it->inputs.end(); ++it2)
		{
			sumInputs += *it2;
		}
		if (it->type == 0) {
			it->output = linear(sumInputs);
		}
		if (it->type == 1) {
			it->output = sigmoid(sumInputs);
		}
	}

	//Delete the old inputs
	for (vector<NodeGene>::iterator it = m_genome.m_NodeGenes.begin(); it != m_genome.m_NodeGenes.end(); ++it)
	{
		it->inputs.clear();
	}

	//Update all inputs in nodes through connections
	for (vector<ConnectionGene>::iterator it = m_genome.m_ConnectionGenes.begin(); it != m_genome.m_ConnectionGenes.end(); ++it)
	{
		double input = 0.0;
		input = m_genome.m_NodeGenes[it->ConnFromNodeNumber].output * it->weight;
		m_genome.m_NodeGenes[it->ConnToNodeNumber].inputs.push_back(input);
	}
}


//Returns the output of the Organism as a vector of doubles
vector<double> Organism::getOutputs()
{
	//Returning the output from the output-nodes
	vector<double> outputs;
	for (vector<NodeGene>::iterator it = m_genome.m_NodeGenes.begin(); it != m_genome.m_NodeGenes.end(); ++it)
	{
		if (it->nodeType == -1)
		{
			outputs.push_back(it->output);
		}
	}

	return outputs;
}


//Print this organisms information about their nodes and connections
void Organism::printInfo(
	int organismNumber)
{
	//Printing the information to the consolde
	system("cls");
	cout << "Organism " << organismNumber << "'s information:\n\n";
	cout << "Fitness: " << m_fitness << endl;
	cout << "Number of nodes: " << m_genome.m_NodeGenes.size() << endl;
	cout << "Number of input-nodes: " << m_genome.m_inputNodes << endl;
	cout << "Number of output-Nodes: " << m_genome.m_outputNodes << endl;
	cout << "Number of hidden-nodes: " << m_genome.m_NodeGenes.size() - m_genome.m_inputNodes - m_genome.m_outputNodes << endl;
	cout << "Number of connections: " << m_genome.m_ConnectionGenes.size() << endl;
	int numberBadConnections = 0;
	int numberGoodConnections = 0;
	for (vector<ConnectionGene>::iterator it = m_genome.m_ConnectionGenes.begin(); it != m_genome.m_ConnectionGenes.end(); it++)
	{
		if (it->weight < 0.0)
			numberBadConnections += 1;
		if (it->weight > 0.0)
			numberGoodConnections += 1;
	}
	cout << "Number of negative connections: " << numberBadConnections << endl;
	cout << "Number of positive connections: " << numberGoodConnections << endl;
	cout << "\nGenome information:\n";
	for (vector<ConnectionGene>::iterator it = m_genome.m_ConnectionGenes.begin(); it != m_genome.m_ConnectionGenes.end(); ++it)
	{
		cout << "From: " << it->ConnFromNodeNumber << "\n";
		cout << "To: " << it->ConnToNodeNumber << "\n";
		cout << "Historical Marking: " << it->historicalNumber << "\n\n";
	}
}


//Returns the highest innovation number of this Species
int Organism::getHighestInnovationNumber()
{
	int highestInnovationNumber = 0;
	for (vector<ConnectionGene>::iterator it = m_genome.m_ConnectionGenes.begin(); it != m_genome.m_ConnectionGenes.end(); ++it)
	{
		if (it->historicalNumber > highestInnovationNumber)
		{
			highestInnovationNumber = it->historicalNumber;
		}
	}
	return highestInnovationNumber;
}


/*THRESHOLD FUNCTIONS*/
//Linear threshold
double Organism::linear(
	const double &sum)
{
	if (sum <= 0.0) { return 0.0; }
	else { return sum; }
}


//Sigmoid threshold
double Organism::sigmoid(
	const double &sum)
{
	return ((1 / (1 + exp(-sum))) - 0.5);
}

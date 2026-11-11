#pragma once
#include <vector>

using namespace std;


//Struct for the gene of a node
struct NodeGene
{
	int nodeType;						//Which layer this node is (0 = input, -1 = output, 1 = hidden)
	int number;							//Identification number for a node
	int type;							//What type of threshold function this is
	vector<double>	inputs = { 0.0 };	//Inputs in this node
	double output = 0.0;						//Output of this node

};


//Struct for the gene of a connection
struct ConnectionGene
{
	int ConnFromNodeNumber;				//Node from which the connection flows from
	int ConnToNodeNumber;				//Node to which the connection flows
	double weight;						//Connection weight between the two nodes
	bool enabled;						//Is this connection enabled or not?
	int historicalNumber;				//Historical marking number for connections
};


//Struct for a genome
struct Genome
{
	//A Vector of Genes for the nodes and the connections
	vector<NodeGene> m_NodeGenes;
	vector<ConnectionGene> m_ConnectionGenes;

	//Data of the Genome
	int m_speciesNumber;
	int m_inputNodes;
	int m_outputNodes;
	double m_fitness;
	int m_highestInnovationNumber;
};
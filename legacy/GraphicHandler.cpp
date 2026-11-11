#include "GraphicHandler.h"



GraphicHandler::GraphicHandler()
{
	//Sets the window size and title
	m_neuralNetworkWindow.create(sf::VideoMode(WIDTH_NETWORK, HEIGHT_NETWORK), "Neural Network of species");
	m_gameWindow.create(sf::VideoMode(WIDTH_GAME, HEIGHT_GAME), "Game");

	//Loads the font from a file
	sf::Font font;
	if (!font.loadFromFile("CurlyLou.ttf"))
	{
		perror("Font file couldnt be loaded.");
	}
	m_font = font;


}


GraphicHandler::~GraphicHandler()
{
}


//Print an organism (TODO: REDO THIS PRINT FUNCTION!)
void GraphicHandler::printOrganism(
	Organism &organism)
{
	//Clear the screen
	sf::Color color(100, 100, 100);
	m_neuralNetworkWindow.clear(color);

	/*Drawing the nodes*/
	//Vector of shapes for the neurons
	vector<sf::CircleShape> neurons;

	//Count the different nodes
	int inputNeuronsNumber = 0;
	int outputNeuronNumber = 0;
	int hiddenNeuronNumber = 0;
	//Because of a compiler bug we have to define a dummy Genome here
	for (vector<NodeGene>::iterator it = organism.getGenome().m_NodeGenes.begin(); it != organism.getGenome().m_NodeGenes.end(); ++it)
	{
		if (it->nodeType == 0)
		{
			inputNeuronsNumber += 1;
		}
		else if (it->nodeType == -1)
		{
			outputNeuronNumber += 1;
		}
		else
		{
			hiddenNeuronNumber += 1;
		}
	}
	int numberInputNeuronsSoFar = 0;
	int numberOutputNeuronsSoFar = 0;
	int numberHiddenNeuronsSoFar = 0;

	//Position and node number is saved in a map for later connection drawing (TODO: Is there maybe a faster code to do this?)
	map<int, vector<double>> nodePositons;

	//Text for the node-output-values
	vector<vector<double>> textVector;

	//Iterating over every Node
	for (vector<NodeGene>::iterator it = organism.getGenome().m_NodeGenes.begin(); it != organism.getGenome().m_NodeGenes.end(); ++it)
	{
		//Add the input neurons and their ouputvalue
		if (it->nodeType == 0)
		{
			//Create the neuron shape
			sf::CircleShape tmp(NODE_RADIUS);
			tmp.setPosition(25.0f, (HEIGHT_NETWORK / inputNeuronsNumber + 1) * numberInputNeuronsSoFar + 2 * NODE_RADIUS);
			tmp.setOutlineThickness(2);
			tmp.setOutlineColor(sf::Color(250, 150, 100));
			if (it->type == 0)
			{
				tmp.setFillColor(sf::Color::Black);
			}
			else
			{
				tmp.setFillColor(sf::Color::Blue);
			}
			neurons.push_back(tmp);
			//Save the neuron shape positon
			vector<double> position = { 25.0f, (HEIGHT_NETWORK / inputNeuronsNumber + 1) * numberInputNeuronsSoFar + 2 * NODE_RADIUS };
			nodePositons[it->number] = position;

			//Save the neuron text positon
			vector<double> text = { 25.0f - 2 * NODE_RADIUS, (HEIGHT_NETWORK / inputNeuronsNumber + 1) * numberInputNeuronsSoFar + 2 * NODE_RADIUS - 2.5*NODE_RADIUS , it->output };
			textVector.push_back(text);
			numberInputNeuronsSoFar += 1;
		}

		//Add the output neurons and their ouputvalue
		else if (it->nodeType == -1)
		{
			//Create the neuron shape
			sf::CircleShape tmp(NODE_RADIUS);
			tmp.setPosition(WIDTH_NETWORK - 25.0f - 2 * NODE_RADIUS , (HEIGHT_NETWORK / outputNeuronNumber + 1) * numberOutputNeuronsSoFar + 2 * NODE_RADIUS);
			tmp.setOutlineThickness(2);
			tmp.setOutlineColor(sf::Color(20, 150, 100));
			if (it->type == 0)
			{
				tmp.setFillColor(sf::Color::Black);
			}
			else
			{
				tmp.setFillColor(sf::Color::Blue);
			}
			neurons.push_back(tmp);
			//Save the neuron shape positon
			vector<double> position = { WIDTH_NETWORK - 25.0f - 2 * NODE_RADIUS , (HEIGHT_NETWORK / outputNeuronNumber + 1) * numberOutputNeuronsSoFar + 2 * NODE_RADIUS };
			nodePositons[it->number] = position;

			//Save the neuron text positon
			vector<double> text = { WIDTH_NETWORK - 25.0f - 2 * NODE_RADIUS - 2 * NODE_RADIUS, (HEIGHT_NETWORK / outputNeuronNumber + 1) * numberOutputNeuronsSoFar + 2 * NODE_RADIUS - 2.5*NODE_RADIUS , it->output };
			textVector.push_back(text);
			numberOutputNeuronsSoFar += 1;
		}

		//Add the deep layer neurons and their ouputvalue
		else
		{
			//Create the neuron shape
			sf::CircleShape tmp(NODE_RADIUS);
			//Set the new node position

			tmp.setPosition(WIDTH_NETWORK/2.0, (HEIGHT_NETWORK / hiddenNeuronNumber) * numberHiddenNeuronsSoFar + 2 * NODE_RADIUS);
			tmp.setOutlineThickness(2);
			tmp.setOutlineColor(sf::Color(100, 20, 100));
			if (it->type == 0)
			{
				tmp.setFillColor(sf::Color::Black);
			}
			else
			{
				tmp.setFillColor(sf::Color::Blue);
			}
			neurons.push_back(tmp);
			//Save the neuron shape positon
			vector<double> position = { WIDTH_NETWORK / 2.0, (HEIGHT_NETWORK / hiddenNeuronNumber) * numberHiddenNeuronsSoFar + 2 * NODE_RADIUS };
			nodePositons[it->number] = position;

			//Save the neuron text positon
			vector<double> text = { WIDTH_NETWORK / 2.0 - 2 * NODE_RADIUS, (HEIGHT_NETWORK / hiddenNeuronNumber) * numberHiddenNeuronsSoFar + 2 * NODE_RADIUS - 2.5*NODE_RADIUS , it->output };
			textVector.push_back(text);
			numberHiddenNeuronsSoFar += 1;
		}
	}

	/*Drawing the connections*/
	//Vector of 2-dimension vertex for the lines
	vector<vector<sf::Vertex>> lines;

	//Map iterator for the 2 vertex positions of the connetion ends
	map<int, vector<double>>::iterator Position1;
	map<int, vector<double>>::iterator Position2;
	//Go through the m_ConnectionGen vector and look for connections that are enabled
	for (vector<ConnectionGene>::iterator it = organism.getGenome().m_ConnectionGenes.begin(); it != organism.getGenome().m_ConnectionGenes.end(); ++it)
	{
		if (it->enabled)
		{
			Position1 = nodePositons.find(it->ConnFromNodeNumber);
			if (Position1 == nodePositons.end())
			{
				perror("Draw error: Connection-end 1 not connected to any Node!");
			}
			Position2 = nodePositons.find(it->ConnToNodeNumber);
			if (Position2 == nodePositons.end())
			{
				perror("Draw error: Connection-end 2 not connected to any Node!");
			}
			vector<sf::Vertex> tmp = { sf::Vector2f(Position1->second[0] + NODE_RADIUS, Position1->second[1] + NODE_RADIUS) , sf::Vector2f(Position2->second[0] + NODE_RADIUS, Position2->second[1] + NODE_RADIUS) };
			if (it->weight < 0)
			{
				tmp[0].color = sf::Color::Red;
				tmp[1].color = sf::Color::Red;
			}
			else
			{
				tmp[0].color = sf::Color::Black;
				tmp[1].color = sf::Color::Black;
			}
			lines.push_back(tmp);
		}
	}

	//Draw the nodes
	for (vector<sf::CircleShape>::iterator it = neurons.begin(); it != neurons.end(); ++it)
	{
		m_neuralNetworkWindow.draw(*it);
	}

	//Draw the connections
	for (vector<vector<sf::Vertex>>::iterator it = lines.begin(); it != lines.end(); ++it)
	{
		//For some reason SFML only takes arrays but no vectors (maybe my fault) so we have to convert the vector back into an array
		sf::Vertex vertexArray[] = { (*it)[0], (*it)[1] };
		vertexArray[0].color = (*it)[0].color;
		vertexArray[1].color = (*it)[1].color;
		m_neuralNetworkWindow.draw(vertexArray, 2, sf::Lines);
	}

	//Draw the outputvalues
	for (vector<vector<double>>::iterator it = textVector.begin(); it != textVector.end(); ++it)
	{
		sf::Text text;
		text.setFont(m_font);
		text.setCharacterSize(16);
		text.setFillColor(sf::Color::Black);
		text.setOutlineColor(sf::Color::Black);
		text.setString(to_string((*it)[2]));
		sf::Vector2f pos((*it)[0], (*it)[1]);
		text.setPosition(pos);
		m_neuralNetworkWindow.draw(text);
	}

	m_neuralNetworkWindow.display();
}


//Prints the game entites to the window
void GraphicHandler::printGame(
	vector<projectile> &projectiles,
	playerStruct &player)
{
	//Clear the screen
	sf::Color color(100, 100, 100);
	m_gameWindow.clear(color);

	//Vector of shapes for the projectiles
	vector<sf::CircleShape> projectileShapes;
	for (vector<projectile>::iterator it = projectiles.begin(); it != projectiles.end(); ++it) {
		sf::CircleShape tmp(PROJECTILE_RADIUS);
		tmp.setFillColor(sf::Color::Red);
		tmp.setPosition(it->positionX, (it->positionY + 0.5) * (HEIGHT_GAME / (NUMBER_ROWS + 1)));
		projectileShapes.push_back(tmp);
	}
	//Draw the projectiles (Better overlook, but 2 loops instead of one!)
	for (vector<sf::CircleShape>::iterator it = projectileShapes.begin(); it != projectileShapes.end(); ++it) {
		m_gameWindow.draw(*it);
	}
	//Draw the player
	sf::CircleShape tmp(PLAYER_RADIUS);
	tmp.setFillColor(sf::Color::Blue);
	tmp.setPosition(player.positionX, (player.positionY + 0.5) * (HEIGHT_GAME / (NUMBER_ROWS + 1)));
	projectileShapes.push_back(tmp);
	m_gameWindow.draw(tmp);
	m_gameWindow.display();
}


//Grabs the events on both windows
void GraphicHandler::getEvents()
{
	sf::Event event;
	while (m_neuralNetworkWindow.pollEvent(event) || m_gameWindow.pollEvent(event))
	{
		if (event.type == sf::Event::Closed) {
			m_neuralNetworkWindow.close();
			m_gameWindow.close();
		}
	}
}
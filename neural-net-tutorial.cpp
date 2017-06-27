/*
 * tutorial from
 * neural-net-tutorial.cpp
 * David Miller, http://millermattson.com/dave
 * See the associated video for instructions: http://vimeo.com/19569529
 * compile with g++
 */
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;
/*then does not require to use std:: below to remember where to take it*/


/**
 * struct Connection - define the weight and deltaweight associated with a
 * neuron
 */
struct Connection
{
	double weight;
	double deltaWeight;
}


class Neuron;

typedef vector<Neuron> Layer;

/* ********************** class Neuron ******************* */

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outputVal = val; }
	double getOutputVla(void) const { return m_outputVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta; /* [0.0..1.0] overall net training rate*/
	static double alpha; /* [0.0..n] multiplier of last weight change (momentum)*/
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double (RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	unsigned m_myIndex;
	vector<Connection> m_outputWeights;
	double m_gradient;
};

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

double Neuron::transferFunction(double x)
{
/* use tanh, values between -1 and 1 */
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
/* use approximation */
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	/* sum the previous layer's outputs (which are our inputs), include the
	 * bias node from the previousl layer
	 */
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() *
			prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outputVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	/* sum our contributions of the errors at the nodes we feed*/
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(Layer nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::tranferFunctionDerivative(m_outputVal);
}

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights(Layer &prevLayer)
{
	/* the weights to be updated are in the connections container in the
	 * neurons in the preceding layer
	 */
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
		double newDeltaWeight =
			/* Individual input, magnified by the gradient and train rate*/
			eta
			* neuron.getOutputVal()
			* m_gradient
			/* also add momentum = a fraction of the preivous delta weight*/
			+ alpha
			* oldDelataWeight;
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

/* ********************** class Net ********************** */
class Net
{
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals);
	void getResults(vector<double> &resultVals) const {}; /*does not modify the object, feels arg*/

private:
	vector<Layer> m_layers; /* m_layers[layerNum][neuronNum]*/
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

Net::Net(const vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++ layerNum)
	{
		/*apprend a a new layer on the container*/
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size - 1 ? 0 :
			topology[layerNum + 1];
		/* we have a new layer, now fill it with neurons and add a
		 * bias neuron to the layer
		 */
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum];
		     ++neuronNum)
		{
			/*back gives access to most recent pushed element*/
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			cout << "Made a Neuron!" << endl; /*print stdout*/
		}
		/* force the bias node'soutput to 1.0 (it was the last neuron
		 * pushed in this layer
		 */
		m_layers.back().back().setOuputVal(1.0);
	}
}

void Net::feedForward(const vector<double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size() - 1);

	/*assign (latch) the input values into the input neurons*/
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][1].setOutputVal(inputVals[i]);
	}

	/* forward propagate */
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const vector<double> &targetVals)
{
	/* calculate overall net error (RMS of output neuron errors)*/
	Layer &outputLayer = m_Layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;
	m_error = sqrt(m_error); /* RMS */

	/* implement a recent average measurement for printing */
	m_recentAverageError =
		(m_recentAverageError + m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

	/* calculate output layer gradients */
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	/* calculate gradients on hidden layers */
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	/* for all layers from outputs to first hidden layer,
	 * update connection weights
	 */
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < hiddenLayer.size() - 1; ++n)
		{
			hiddenLyer[n].updateInputWeights(prevLayer);
		}
	}
}

void Net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() -1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}


int main()
{
	/* eg: topology = {3, 2, 1} 3 layers: 3 input nodes, 2 hidden nodes in
	 * 1 hidden layer and 1 output node in output layer
	 */
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	/*variable size array*/
	std::vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);
}

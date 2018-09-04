
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>

#include "net.hpp"
#include "neuron.hpp"






/* NET IMPLEMENTATION ****************************************************************************/

double Net::m_recentAverageSmoothingFactor = 1.0;

Net::Net(std::vector<unsigned>& topology){

	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0 ; layerNum < numLayers ; ++layerNum){
		m_layers.push_back(Layer());

		unsigned numOutputs = (layerNum == numLayers - 1)? 0 : topology[layerNum + 1];


		for(unsigned j = 0 ; j < topology[layerNum] ; ++j){
			m_layers.back().push_back(Neuron(numOutputs));
		}

		// A Bias neuron is needed in each layer
		m_layers.back().push_back(Neuron(numOutputs));

		// Force the Bias neuron's output val to 1.0
		m_layers.back().back().setOutputVal(1.0);
	}

}

void Net::feedForward(std::vector<double> const& inputVals){

	// We make sure that the num of inputs is the same as the num of neurons in input layer
	assert(inputVals.size() == m_layers[0].size() - 1);		// The Bias neuron is not included

	for(unsigned n = 0 ; n < inputVals.size() ; ++n){
		m_layers[0][n].setOutputVal(inputVals[n]);
	}

	for(unsigned layerNum = 1 ; layerNum < m_layers.size() ; ++layerNum){
		for(unsigned n = 0 ; n < m_layers[layerNum].size() - 1 ; ++n){		// The -1 is to avoid the Bias n
			Layer& prevLayer = m_layers[layerNum - 1];
			m_layers[layerNum][n].feedForward(prevLayer, n);
		}
	}
}

void Net::backPropagation(std::vector<double> const& targetVals){

	// This method calculates the errors, and tries to correct these errors

	// Calculating the overall neural net error (The RMS val of neurons output error)
	Layer& outputLayer = m_layers.back();
	m_rmsError = 0.0;

	for(unsigned n = 0 ; n < outputLayer.size() - 1 ; ++n){	// The -1 is to avoid the Bias neuron
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		m_rmsError += delta * delta;
	}

	m_rmsError /= (outputLayer.size() - 1);
	m_rmsError = sqrt(m_rmsError);


	// Implementing a recent average measurement (to give a running indication of how well the training is going)
	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_rmsError)
		/ (m_recentAverageSmoothingFactor + 1.0);


	// Calculate the output layer gradient
	for(unsigned n = 0 ; n < outputLayer.size() ; ++n){
		outputLayer[n].calculateOutputGradients(targetVals[n]);
	}


	// Calculate the hidden layers gradients
	for(unsigned layerNum = m_layers.size() - 2 ; layerNum > 0 ; --layerNum){
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];

		for(unsigned n = 0 ; n < hiddenLayer.size() ; ++n){
			hiddenLayer[n].calculateHiddenGradients(nextLayer);
		}
	}


	// Update connections weight from output layer to 1st hidden layer
	for(unsigned layerNum = m_layers.size() - 1 ; layerNum > 0 ; --layerNum){
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for(unsigned n = 0 ; n < layer.size() - 1 ; ++n){		// Bias neuron is omitted 
			layer[n].updateInputWeights(prevLayer, n);
		}
	}
}

void Net::getResults(std::vector<double> & resultVals) const{
	resultVals.clear();

	for(unsigned n = 0 ; n < m_layers.back().size() - 1 ; ++n){
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
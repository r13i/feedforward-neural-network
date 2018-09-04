
#include <cstdlib>
#include <vector>
#include <cmath>

#include "neuron.hpp"





/* NEURON IMPLEMENTATION *************************************************************************/

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

double Neuron::transfertFunc(double val){ return tanh(val); }
double Neuron::randomWeight(void){ return rand() / double(RAND_MAX); }

Neuron::Neuron(unsigned numOutputs){

	for(unsigned connec = 0 ; connec < numOutputs ; ++connec){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = Neuron::randomWeight();
	}
}

void Neuron::feedForward(Layer const& prevLayer, unsigned ownIndex){

	// Calculation of the average of the values in the previous layer
	double sum = 0.0;

	for(unsigned n = 0 ; n < prevLayer.size() ; ++n){
		sum += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[ownIndex].weight;	// No need for getters since this is a member func
	}

	m_outputVal = Neuron::transfertFunc(sum);	// Implements a sigmoid math op (e.g. tanh)
}

double Neuron::sumDOW(Layer const& nextLayer) const{
	// Sum the contributions of the errors at the neurons feeded by the actual neuron
	double sum = 0.0;

	for(unsigned n = 0 ; n < nextLayer.size() - 1 ; ++n){
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calculateOutputGradients(double targetVal){
	double delta = targetVal - getOutputVal();
	m_gradient = delta * transfertFuncDerivative(getOutputVal());
}

void Neuron::calculateHiddenGradients(Layer const& nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient = dow * transfertFuncDerivative(getOutputVal());
}

void Neuron::updateInputWeights(Layer& prevLayer, unsigned ownIndex){
	// The weights to update are contained in the Connection container in the neurons of prev layer
	for(unsigned n = 0 ; n < prevLayer.size() ; ++n){		// The Bias is actually included

		Neuron& neuron = prevLayer[n];

		double oldDeltaWeight = neuron.m_outputWeights[ownIndex].deltaWeight;
		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate (eta = 0.0 slow learner; 0.2 medium learner; 1.0 reckless learner)
			eta
			* neuron.getOutputVal()
			* m_gradient

			// Plus a momentum = fraction of the previous delta weight (alpha = 0.0 no momentum; 0.5 moderate momentum)
			+
			alpha
			* oldDeltaWeight;


		neuron.m_outputWeights[ownIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[ownIndex].weight += newDeltaWeight;
	}
}
#ifndef NEURON_H
#define NEURON_H





/*	WEIGHTED CONNECTIONS *************************************************************************/
struct Connection{
	double weight;
	double deltaWeight;
};





/*	CLASS NEURON *********************************************************************************/
class Neuron;





/*	DEF OF A NEURONS LAYER ***********************************************************************/
typedef std::vector<Neuron> Layer;





class Neuron{
public:
	// Constructor
	Neuron(unsigned numOutputs);

	// Getters, Setters
	void setOutputVal(double val){ m_outputVal = val; }
	double getOutputVal(void) const { return m_outputVal; }

	// Other methods
	void feedForward(Layer const& prevLayer, unsigned ownIndex);
	double sumDOW(Layer const& nextLayer) const;
	void calculateOutputGradients(double targetVal);
	void calculateHiddenGradients(Layer const& nextLayer);
	void updateInputWeights(Layer& prevLayer, unsigned ownIndex);

private:
	static double eta;		// [0.0 .. 1.0] The overall net training rate
	static double alpha;	// [0.0 .. n]	Momentum: Multiplier of last weight change
	static double randomWeight(void);	// P.S.: "static" means this func can be accessed without any instance of the class
	static double transfertFunc(double val);
	static double transfertFuncDerivative(double val){ return 1.0 - val * val;	/* This a Limited Dev of the Derivative if "tanh() */ }

	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	double m_gradient;
};






#endif /*NEURON_H*/
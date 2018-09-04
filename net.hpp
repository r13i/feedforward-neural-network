#ifndef NET_H
#define NET_H


#include "neuron.hpp"


/* See the Net::m_recentAverageSmoothingFactor */


/* CLASS NET *************************************************************************************/
class Net{
public:
	Net(std::vector<unsigned> & topology);

	void feedForward(std::vector<double> const& inputVals);
	void backPropagation(std::vector<double> const& targetVals);
	void getResults(std::vector<double> & resultVals) const;

	double getRecentAverageError(void) const { return m_recentAverageError; }
	
private:
	std::vector<Layer> m_layers;
	double m_rmsError;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};



#endif /*NET_H*/
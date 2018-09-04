/*
*	Author: Achouri A.Redouane
*
*	Resource:
*		https://vimeo.com/19569529 or https://www.youtube.com/watch?v=KkwX7FkLfug
*/



#include <iostream>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>

#include "net.hpp"

using namespace std;





/* TRAINING DATA CLASS ***************************************************************************/
class TrainingData{
public:
	TrainingData(string const& filename);
	~TrainingData();

	bool isEof(void){ return m_trainingDataFile.eof(); }
	void readTopology(vector<unsigned>& topology);

	unsigned readNextInputs(vector<double>& inputVals);
	unsigned readTargetOutputs(vector<double>& targetVals);


private:
	ifstream m_trainingDataFile;
};





/*************************************************************************************************/
void printVect(vector<double> const& v){

	for(unsigned i = 0 ; i < v.size() ; ++i){
		cout <<v[i] <<" ";
	}
	cout <<endl;
}
/*************************************************************************************************/
int main(int argc, char const *argv[]){
	if(argc != 2){
		cout <<endl <<"How to:" <<endl
			<<"\t./this_exe TRAINING_DATA_FILE.txt" <<endl;
		return -1;
	}

	TrainingData trainingData(argv[1]);


	vector<unsigned> topology;		// An example of topology could be {3 input neurons}{2 neurons in 1 hidden layer}{1 output neuron}
	trainingData.readTopology(topology);

	Net myNet(topology);

	vector<double> inputVals, targetVals, resultVals;
	unsigned trainingPass = 0;	// The index of actual training

	while(! trainingData.isEof()){

		// Get new training data and feed it forward
		if(trainingData.readNextInputs(inputVals) != topology[0]){
			break;
		}


		cout <<"Pass: " <<(++trainingPass) <<endl;

		cout <<"Inputs: ";	printVect(inputVals);
		myNet.feedForward(inputVals);

		// Get actual results
		myNet.getResults(resultVals);
		cout <<"Outputs: ";		printVect(resultVals);

		// Train the net what the outputs should have been
		trainingData.readTargetOutputs(targetVals);
		cout <<"Targets: ";		printVect(targetVals);

		assert(targetVals.size() == topology.back());
		
		myNet.backPropagation(targetVals);


		// Report how well the traning is working
		cout <<"Net recent average error: " <<myNet.getRecentAverageError() <<endl;

		cout <<endl;
	}

	cout <<endl <<"Done." <<endl;

	return 0;
}





/* TRAINING DATA IMPLEMENTATION ******************************************************************/
TrainingData::TrainingData(string const& filename){
	m_trainingDataFile.open(filename.c_str());
}

TrainingData::~TrainingData(){
	m_trainingDataFile.close();
}

void TrainingData::readTopology(vector<unsigned>& topology){
	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);

	ss >>label;

	if(this->isEof() || label.compare("Topology:") != 0){
		cout <<endl <<"Training data file is empty OR Cannot find actual topology. Abort." <<endl;
		abort();
	}

	while(! ss.eof()){
		unsigned n;
		ss >>n;
		topology.push_back(n);
	}

	return;
}

unsigned TrainingData::readNextInputs(vector<double>& inputVals){
	inputVals.clear();

	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);

	ss >>label;
	if(label.compare("In:") == 0){
		double val;
		while(ss >>val){
			inputVals.push_back(val);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::readTargetOutputs(vector<double>& targetVals){
	targetVals.clear();

	string line;
	string label;

	getline(m_trainingDataFile, line);
	stringstream ss(line);

	ss >>label;
	if(label.compare("Out:") == 0){
		double val;
		while(ss >>val){
			targetVals.push_back(val);
		}
	}

	return targetVals.size();
}
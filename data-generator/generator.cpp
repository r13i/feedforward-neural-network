#include <iostream>
#include <cstdlib>
#include <sstream>
#include <string>

using namespace std;

/*
* Keep the structure:
*	Topology: val1 val2 ... val_n
*	In: x x x ... (with a length of val1)
*	Out: y y y ... (with a length of val2)
*/

int main(int argc, char* argv[]){
	if(argc != 2){
		cout <<endl <<"How to:" <<endl
			<<"\t./this_exe NUM_OF_SAMPLES" <<endl;
		return -1;
	}

	string argVal(argv[1]);
	stringstream ss(argVal);

	unsigned samplesCount;
	ss >>samplesCount;

	cout <<"Topology: 2 4 1" <<endl;	// 2 inputs and 1 output

	// This is actually a XOR gate
	for(unsigned short i = 0 ; i < samplesCount ; ++i){
		short n1 = (short)(2.0 * rand() / double(RAND_MAX));
		short n2 = (short)(2.0 * rand() / double(RAND_MAX));
		short res = n1 ^ n2;

		cout <<"In: " <<n1 <<".0 " <<n2 <<".0" <<endl;
		cout <<"Out: " <<res <<".0" <<endl;
	}
}
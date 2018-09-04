nn: main.o net.o neuron.o
	g++ -o nn main.o net.o neuron.o

main.o: main.cpp net.hpp
	g++ -c main.cpp

net.o: net.hpp net.cpp neuron.hpp neuron.cpp
	g++ -c net.cpp

neuron.o: neuron.hpp neuron.cpp
	g++ -c neuron.cpp

clean:
	rm -rf *.o
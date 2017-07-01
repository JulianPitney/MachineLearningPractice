#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>

using namespace std;

class SigmoidNeuron 
{

public:
	vector<float> inputWeights;
	vector<float> inputValues;
	double bias;

	// This is defined/computed by the sigmoid function (3,4 from tutorial)
	double output;


	float calculateOutput();

};


// Uses sigmoid function to calculate output value of neuron
float SigmoidNeuron::calculateOutput() {

	float z = 0;
	
	// Dot product of vectors of weights and inputs
	for(unsigned int i = 0; i < inputWeights.size(); i++)
	{
		z += inputWeights[i] * inputValues[i];
	}

	z += bias;
	
	// Sigmoid function
	output = (1) / (1 + exp(-z)); 
	return output;
}


int main(int argc, char** argv)
{
	
	SigmoidNeuron *testNeuron;

	for(int x = 0; x < 100; x++)
	{
		testNeuron = new SigmoidNeuron;
		
	// Generate random inputs, inputWeights and bias for neuron
	for(int i = 0; i < 10; i++)
	{
		// inputWeights can be negative or positive
		float input = (float) (rand() % 257) / (256);
		if(rand() % 101 < 50)
		{
			input = input*-1;
		}
		testNeuron->inputWeights.push_back(input);
	
		// input can only be positive (we're using 0-256 since we'll be using 
		// greyscale pixel values later
		input = (float) (rand() % 257) / (256);
		testNeuron->inputValues.push_back(input);	
	
		// Bias can be negative or positive
		input = rand() % 5;
		if (rand() % 101 < 50)
		{
			input = input*-1;
		}
		testNeuron->bias = input;
	}


	cout << "Neuron Output: " <<  testNeuron->calculateOutput() << endl;

	}
	return 0;
}

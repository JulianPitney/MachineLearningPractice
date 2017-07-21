#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<cv.h>
#include<highgui.h>
#include<random>

using namespace std;
using namespace cv;

struct imgDimensions {

	int rows;
	int cols;
	int channels;
};


Mat getImage(char* imgPath) {

	Mat img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	
	if(!img.data)
	{
		cout << "Image is empty\n";
	}
	
	return img;
}


struct imgDimensions getImgDimensions(Mat img) {

	struct imgDimensions output;

	output.rows = img.rows;
	output.cols = img.cols;
	output.channels = img.channels();

	return output;
}




class SigmoidNeuron 
{

public:
	vector<float> inputWeights; // Between 0-1 (not sure)
	vector<float> inputValues; // Between 0-1 (sure)
	double bias; // Between 0-1 (not sure)

	// This is defined/computed by the sigmoid function and can take any value from 0 to 1
	float output = 0;
	// Input to sigmoid function (Need to save this each time we calculate neuron's output as it's used in
	// calculating gradients for backpropagation
	float sigmoidInput_z = 0;
	void collectInputs(vector<SigmoidNeuron*>& previousLayer);
	float calculateOutput();

};


void SigmoidNeuron::collectInputs(vector<SigmoidNeuron*>& previousLayer) {

	for(unsigned int i = 0; i < previousLayer.size(); i++)
	{
		inputValues.push_back(previousLayer[i]->output);
	}
}


// Uses sigmoid function to calculate output value of neuron
float SigmoidNeuron::calculateOutput() {

	float z = 0;
	
	// Dot product of vectors of weights and inputs
	for(unsigned int i = 0; i < inputWeights.size(); i++)
	{
		z += inputWeights[i] * inputValues[i];
	}

	z -= bias;

	// Saving sigmoid input for gradient descent calculations later on
	sigmoidInput_z = z;

	// Sigmoid function
	output = (1.00) / (1.00 + exp(-z)); 
	
	return output;
}



class NeuralNetwork
{

public:

	// Network Layers
	vector<SigmoidNeuron*> inputLayer;
	vector<SigmoidNeuron*> hiddenLayer;
	vector<SigmoidNeuron*> outputLayer;

	int targetValue;

	// Gradient Values
	vector<float> outputLayerWeightGradients;
	vector<float> outputLayerBiasGradients;
	vector<float> hiddenLayerWeightGradients;
	vector<float> hiddenLayerBiasGradients;
	
	void feedInputLayer(Mat img);
	void setTargetValue(int targetValue);
	void setDefaultWeights();
	void updateInputLayer();
	void updateHiddenLayer();
	void updateOutputLayer();
	void purgeHiddenLayer();
	void purgeOutputLayer();
	void populateInputLayer(struct imgDimensions dims);
	void populateHiddenLayer(int hLayerSize);
	void populateOutputLayer(int oLayerSize);
	void fireNeuralNetwork(Mat inputImage, int targetValue);
	
	void calculateOutputLayerGradients();
};

// Populates layers with default weights and Biases (must be called after populating
// all layers (using populateXXXXXX functions)
void NeuralNetwork::setDefaultWeights() {

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		for(unsigned int x = 0; x < inputLayer.size(); x++)
		{
			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			hiddenLayer[i]->inputWeights.push_back(r);
		}	
		hiddenLayer[i]->bias = (float) (rand() % inputLayer.size() + 1);
	}

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		for ( unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			outputLayer[i]->inputWeights.push_back(r);
		}
		outputLayer[i]->bias = (float) (rand() % hiddenLayer.size() + 1);
	}	
}

void NeuralNetwork::feedInputLayer(Mat img) {

	struct imgDimensions dims = getImgDimensions(img);	

	uchar* p = img.data;
	int currentNeuron = 0;

	for(int i = 0; i < dims.rows; i++)
	{
		p = img.ptr<uchar>(i);
		for(int x = 0; x< dims.cols; x++)
		{
			inputLayer[currentNeuron]->output = (float)p[x]/255.00;
			currentNeuron++;
		}
	}		

}


// Target values start at 0 (e.g if target is 1 then 0th node in outputLayer should be 1, so set target to 0)
void NeuralNetwork::setTargetValue(int targetValue) {

	this->targetValue = targetValue;
}

void NeuralNetwork::updateHiddenLayer() {

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayer[i]->collectInputs(inputLayer);
		hiddenLayer[i]->calculateOutput();
	}
}

void NeuralNetwork::purgeHiddenLayer() {

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayer[i]->inputValues.clear();	
	}
}

void NeuralNetwork::updateOutputLayer() {

	for(unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i]->collectInputs(hiddenLayer);
		outputLayer[i]->calculateOutput();
	}
}

void NeuralNetwork::purgeOutputLayer() {

	for(unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i]->inputValues.clear();
	}

}

void NeuralNetwork::populateInputLayer(struct imgDimensions dims) {

	for(int i = 0; i < dims.rows*dims.cols; i++)
	{
		this->inputLayer.push_back(new SigmoidNeuron);
	}

}

void NeuralNetwork::populateHiddenLayer(int hLayerSize) {

	for(int i = 0; i < hLayerSize; i++)
	{
		this->hiddenLayer.push_back(new SigmoidNeuron);
	}
}

void NeuralNetwork::populateOutputLayer(int oLayerSize) {

	for(int i = 0; i < oLayerSize; i++)
	{
		this->outputLayer.push_back(new SigmoidNeuron);
	}
}

void NeuralNetwork::fireNeuralNetwork(Mat inputImage, int targetValue) {

	purgeHiddenLayer();
	purgeOutputLayer();
	feedInputLayer(inputImage);
	setTargetValue(targetValue);
	updateHiddenLayer();
	updateOutputLayer();

}


// This seems to be working okay, but the hidden layer is routinely outputting 0's,
// which kinda fucks up the gradient calculations. There may be something wrong with
// the pipeline leading up to calculating hidden layer node outputs. Need to look into
// why/how those nodes are outputting 0.
void NeuralNetwork::calculateOutputLayerGradients() {

	float targetValues[outputLayer.size()];
	float actualValues[outputLayer.size()];
	float outputLayerInputs_z[outputLayer.size()];
	float hiddenLayerOutputs[hiddenLayer.size()];

	// Collecting values 
	for(unsigned int i = 0; i < outputLayer.size(); i++)
	{
		if (i == targetValue)
		{
			targetValues[i] = (float) 1.000;
		}
		else
		{
			targetValues[i] = (float) 0.000;
		}

		actualValues[i] = outputLayer[i]->output;
		outputLayerInputs_z[i] = outputLayer[i]->sigmoidInput_z;
	}

	// Collecting more values
	for(unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayerOutputs[i] = hiddenLayer[i]->output;
	}

	
	// Calculate Output Layer Gradients
	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		float tempBiasGradient;
		float AV = actualValues[i];
		float TV = targetValues[i];
		float z = outputLayerInputs_z[i];
		// Output layer bias gradient for node i of output layer
		tempBiasGradient = (AV - TV) * (exp(z) / (exp(z) + 1)*(exp(z)+ 1));
		outputLayerBiasGradients.push_back(tempBiasGradient);
		cout << "BIas gradient=" << tempBiasGradient << endl;

		for (unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float a = hiddenLayerOutputs[x];
			float tempWeightGradient = tempBiasGradient;
			// Output layer weight gradient for weight connecting node x of hidden layer to 
			// node i of output layer
			tempWeightGradient *= a;
			outputLayerWeightGradients.push_back(tempWeightGradient);	
			cout << "Weight gradient=" << tempWeightGradient << endl;
		}
	}




}

	
int main(int argc, char** argv)
{

	// Create and Init neural network
	NeuralNetwork test;
	Mat img = getImage(argv[1]);
	struct imgDimensions dims = getImgDimensions(img);
	test.populateInputLayer(dims);
	test.populateHiddenLayer(15);
	test.populateOutputLayer(10);
	test.setDefaultWeights();


	test.fireNeuralNetwork(img,6);
	
	test.calculateOutputLayerGradients();
	return 0;
}

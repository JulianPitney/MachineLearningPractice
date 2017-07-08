#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<cv.h>
#include<highgui.h>

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



class NeuralNetwork
{

public:

	vector<SigmoidNeuron*> inputLayer;
	vector<SigmoidNeuron*> hiddenLayer;
	vector<SigmoidNeuron*> outputLayer;


	void feedInputLayer(Mat img);
	void populateInputLayer(struct imgDimensions dims);
	void populateHiddenLayer(int hLayerSize);
	void populateOutputLayer(int oLayerSize);
};

void NeuralNetwork::feedInputLayer(Mat img) {

	struct imgDimensions dims = getImgDimensions(img);	

	uchar* p = img.data;
	int currentNeuron = 0;

	for(int i = 0; i < dims.rows; i++)
	{
		p = img.ptr<uchar>(i);
		for(int x = 0; x< dims.cols; x++)
		{
			inputLayer[currentNeuron]->inputValues.push_back((float)p[x]/255.00);
			currentNeuron++;
		}
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


int main(int argc, char** argv)
{

	NeuralNetwork test;
	Mat img = getImage(argv[1]);
	struct imgDimensions dims = getImgDimensions(img);
	
	test.populateInputLayer(dims);
	test.populateHiddenLayer(15);
	test.populateOutputLayer(10);

	// Connect All layers (

	test.feedInputLayer(img);

	return 0;
}

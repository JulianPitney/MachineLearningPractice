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


imgDimensions getImgDimensions(Mat img) {

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


	bool feedInputLayer(char* imgPath);
};

bool NeuralNetwork::feedInputLayer(char* imgPath) {

	Mat img = getImage(imgPath);
	int channels = img.channels();
	int rows = img.rows;
	int cols = img.cols * channels;
	
	uchar* p = img.data;
	int currentNeuron = 0;

	for(int i = 0; i < rows; i++)
	{
		p = img.ptr<uchar>(i);
		for(int x = 0; x< cols; x++)
		{
			inputLayer[currentNeuron]->inputValues.push_back((float)p[x]/255.00);
			currentNeuron++;
		}
	}		

}


int main(int argc, char** argv)
{

	NeuralNetwork test;
	Mat img = getImage(argv[1]);
	struct imgDimensions dims = getImgDimensions(img);
	int cols = dims.cols;
	int rows = dims.rows;
	int channels = dims.channels;
	
	// Populate input layer	
	for(int i = 0; i < rows*cols; i++)
	{
		test.inputLayer.push_back(new SigmoidNeuron);
	}


	// Populate hidden layer(s)
	
	// Populate output layer
	for(int i = 0; i < 10; i++)
	{
		test.outputLayer.push_back(new SigmoidNeuron);
	}

	// Connect All layers (

	// Feed input layer with image
	test.feedInputLayer(argv[1]);

	
	cout << "Expected dimensions: W="<<img.cols << "H=" << img.rows << "C=" << img.channels() << endl;
	cout << "Actual dimensions: W="<<cols << "H=" << rows << "C=" << channels << endl;


	return 0;
}

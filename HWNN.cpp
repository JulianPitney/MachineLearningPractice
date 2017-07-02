#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<cv.h>
#include<highgui.h>

using namespace std;
using namespace cv;



Mat getImage(char* imgPath) {

	Mat img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);
	
	if(!img.data)
	{
		cout << "Image is empty\n";
	}
	
	return img;
}



class SigmoidNeuron 
{

public:
	vector<float> inputWeights; // Between 0-1 (not sure)
	vector<float> inputValues; // Between 0-1 (sure)
	double bias; // Between 0-1 (not sure)

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

	for(int i = 0; i < rows; i++)
	{
		p = img.ptr<uchar>(i);
		for(int x = 0; x< cols; x++)
		{
			inputLayer[x]->inputValues.push_back((float) p[x]);
			cout << "p[x]=" << (float) p[x] << endl;
		}
	}		

}


int main(int argc, char** argv)
{

	NeuralNetwork test;
	
	Mat img = getImage(argv[1]);
	int channels = img.channels();
	int rows = img.rows;
	int cols = img.cols;

	cout << "Channels=" << channels << endl;
	cout << "Rows=" << rows << endl;
	cout << "Cols=" << cols << endl;
	
	for(int i = 0; i < rows; i++)
	{

		for(int x = 0; x < cols; x++)
		{
			test.inputLayer.push_back(new SigmoidNeuron);
		}

	}
	

	test.feedInputLayer(argv[1]);




	for(int i = 0; i< test.inputLayer.size(); i++)
	{
		cout << test.inputLayer[i]->inputValues[0] << endl;
	}




	return 0;
}

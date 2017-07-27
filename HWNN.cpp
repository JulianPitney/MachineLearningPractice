#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<cv.h>
#include<highgui.h>
#include<random>
#include<fstream>


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

	// These need to be purged after each time network runs (very poor design choice....should fix that later)
	vector<float> inputValues; // Between 0-1 (sure)
	float bias;

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

	// Temp hack to deal with z exploding when output of inputLayer is too high
	if (z > 10)
	{
		z = 10; 
	}


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

	float targetValue;

	// Purge these after each training run
	vector<float> outputLayerWeightGradients;
	vector<float> outputLayerBiasGradients;
	vector<float> hiddenLayerWeightGradients;
	vector<float> hiddenLayerBiasGradients;

	float learningRate = 0;

	// Also purge these after each training run
	vector<float> outputLayerWeightDeltaVs;
	vector<float> outputLayerBiasDeltaVs;
	vector<float> hiddenLayerWeightDeltaVs;
	vector<float> hiddenLayerBiasDeltaVs;

	
	void feedInputLayer(Mat img);
	void setTargetValue(float targetValue);
	void setDefaultWeights();
	void updateInputLayer();
	void updateHiddenLayer();
	void updateOutputLayer();
	void purgeHiddenLayer();
	void purgeOutputLayer();
	void populateInputLayer(struct imgDimensions dims);
	void populateHiddenLayer(int hLayerSize);
	void populateOutputLayer(int oLayerSize);
	void fireNeuralNetwork(Mat inputImage, float targetValue);
	
	void calculateOutputLayerGradients();
	void calculateHiddenLayerGradients();
	void setLearningRate(float learningRate);
	void computeDeltaVs();
	void UpdateNetworkVariables();
	void purgeGradientDescentVectors();
	void trainNetwork(Mat trainingInput, float trainingLabel);
	void trainNetwork_MNIST(int trainingSetSize, unsigned char* trainingImages, unsigned char* trainingLabels);
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

		float r3 = 40.00000000 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(50.0000000-40.0000000)));
		hiddenLayer[i]->bias = r3;
	}

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		for ( unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
			outputLayer[i]->inputWeights.push_back(r);
		}

		float r3 = 1.000000000 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(15.00000000-1.00000000)));	
		outputLayer[i]->bias = r3;
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
void NeuralNetwork::setTargetValue(float targetValue) {

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

void NeuralNetwork::fireNeuralNetwork(Mat inputImage, float targetValue) {

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

		for (unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float a = hiddenLayerOutputs[x];
			float tempWeightGradient = tempBiasGradient;
			// Output layer weight gradient for weight connecting node x of hidden layer to 
			// node i of output layer
			tempWeightGradient *= a;
			outputLayerWeightGradients.push_back(tempWeightGradient);
		}
	}




}

void NeuralNetwork::calculateHiddenLayerGradients() {

	float inputLayerOutput_NodeI[inputLayer.size()]; // a sub i
	float hiddenLayerInputs_z[hiddenLayer.size()]; // z sub j
	float outputLayerOutputs[outputLayer.size()]; // a sub k
	float targetOutput[outputLayer.size()];	// t sub k
	float outputLayerInputs_z[outputLayer.size()]; // z sub k
	float hiddenLayertoOutputLayerWeights[outputLayer.size()][hiddenLayer.size()]; // w sub jk


	// sets a sub i
	for(unsigned int i = 0; i < inputLayer.size(); i++)
	{

		inputLayerOutput_NodeI[i] = inputLayer[i]->output;
	}
	
	// sets z sub j
	for(unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayerInputs_z[i] = hiddenLayer[i]->sigmoidInput_z;
	}
	
	// Sets a sub k && t sub k && z sub k	
	for(unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayerOutputs[i] = outputLayer[i]->output;
		outputLayerInputs_z[i] = outputLayer[i]->sigmoidInput_z;	
		if (i == targetValue)
		{
			targetOutput[i] = (float) 1.000;
		}
		else
		{
			targetOutput[i] = (float) 0.000;
	
		}
	}

	// Sets w sub jk
	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		for(unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			hiddenLayertoOutputLayerWeights[i][x] = outputLayer[i]->inputWeights[x];
		}	
	}



	// Start gradient calculations
	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		int z_temp = hiddenLayerInputs_z[i];
		float term1 = exp(z_temp);
		float term2 = (1 + term1) * (1 + term1);
		float gPrimeOfJ = term1/term2;

		float dSubK;
		float Sum = 0;

		for (unsigned int x = 0; x < outputLayer.size(); x++)
		{
			dSubK = (outputLayerOutputs[x] - targetOutput[x]) * (( exp(outputLayerInputs_z[x])) / (1 + exp(outputLayerInputs_z[x]) * exp(outputLayerInputs_z[x])));
			dSubK *= hiddenLayertoOutputLayerWeights[x][i];
			Sum += dSubK;				
			
		}


		float biasGradientTemp = gPrimeOfJ * Sum;
		this->hiddenLayerBiasGradients.push_back(biasGradientTemp);

		for(unsigned int x = 0; x < inputLayer.size(); x++)
		{
			float aSubI = inputLayerOutput_NodeI[x];
			float weightGradientTemp = biasGradientTemp * aSubI;
			hiddenLayerWeightGradients.push_back(weightGradientTemp);
		}


	} 

}

void NeuralNetwork::setLearningRate(float learningRate) {

	this->learningRate = learningRate;
}


// These need to be purged after each input
void NeuralNetwork::computeDeltaVs() {

	for (unsigned int i = 0; i < outputLayerBiasGradients.size(); i++)
	{
		float tempDeltaV = (-1)*(learningRate)*(outputLayerBiasGradients[i]);
		outputLayerBiasDeltaVs.push_back(tempDeltaV);
	}	

	for (unsigned int i = 0; i < outputLayerWeightGradients.size(); i++)
	{
		float tempDeltaV = (-1)*(learningRate)*(outputLayerWeightGradients[i]);
		outputLayerWeightDeltaVs.push_back(tempDeltaV);
	}

	for (unsigned int i = 0; i < hiddenLayerBiasGradients.size(); i++)
	{
		float tempDeltaV = (-1)*(learningRate)*(hiddenLayerBiasGradients[i]);
		hiddenLayerBiasDeltaVs.push_back(tempDeltaV);
	}

	for (unsigned int i = 0; i < hiddenLayerWeightGradients.size(); i++)
	{
		float tempDeltaV = (-1)*(learningRate)*(hiddenLayerWeightGradients[i]);
		hiddenLayerWeightDeltaVs.push_back(tempDeltaV);
	}
}


void NeuralNetwork::UpdateNetworkVariables() {

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayer[i]->bias = (hiddenLayer[i]->bias) - (hiddenLayerBiasDeltaVs[i]);	
		int inputWeightIndexForCurrentNeuron = 0;

		for (unsigned int x = inputLayer.size() * i; x < (inputLayer.size()) * (i + 1); x++)
		{
			hiddenLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron] = (hiddenLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron]) - (hiddenLayerWeightDeltaVs[x]);
			inputWeightIndexForCurrentNeuron++;
		}
	}

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i]->bias = (outputLayer[i]->bias) - (outputLayerBiasDeltaVs[i]);
		int inputWeightIndexForCurrentNeuron = 0;

		for (unsigned int x = hiddenLayer.size() * i; x < (hiddenLayer.size()) * (i + 1); x++)
		{
			outputLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron] = (outputLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron]) - (outputLayerWeightGradients[x]);
			inputWeightIndexForCurrentNeuron++;
		}
	}

}


void NeuralNetwork::purgeGradientDescentVectors() {

	outputLayerWeightGradients.clear();
	outputLayerBiasGradients.clear();
	hiddenLayerWeightGradients.clear();
	hiddenLayerBiasGradients.clear();
	outputLayerWeightDeltaVs.clear();
	outputLayerBiasDeltaVs.clear();
	hiddenLayerWeightDeltaVs.clear();
	hiddenLayerBiasDeltaVs.clear();
}

// bitwise black magic
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char* read_mnist_labels(const char* full_path) {

	ifstream file(full_path);
	unsigned char* labels;


	if (file.is_open())
	{

		int magic_number=0;
		int number_of_labels=0;

		file.read((char*)&magic_number,sizeof(magic_number));
		magic_number = reverseInt(magic_number);
	
		file.read((char*)&number_of_labels,sizeof(number_of_labels));
		number_of_labels = reverseInt(number_of_labels);

		labels = new unsigned char[number_of_labels];

		for(unsigned int i = 0; i < number_of_labels; i++)
		{
			unsigned char temp=0;
			file.read((char*)&temp,sizeof(temp));
			labels[i] = temp;
		}
	}

	return labels;

}

// Reads and loads mnist data set into memory
unsigned char* read_mnist_images(const char* full_path)
{
    ifstream file (full_path);
    unsigned char* images;

    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);

        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);

        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

	images = new unsigned char[number_of_images * 784];
    	int imagesCounter = 0;

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
		    images[imagesCounter] = temp;
		    imagesCounter++;
                }
            }
        }
    }
    return images;
}

// Takes array containing mnist images and starting position of desired image in that array,
// and returns Mat containing desired image
void dispense_mnist_image(int* imgCounter, unsigned char* images, Mat imgDest) {

	uchar* p = imgDest.data;

	for(unsigned int row = 0; row < 28; row++)
	{
		p = imgDest.ptr<uchar>(row);

		for(unsigned int col = 0; col < 28; col++)
		{
			p[col] = images[*imgCounter];
			*imgCounter = *imgCounter + 1;
		}	
	}
}
	
void dispense_mnist_label(int* labelCounter, unsigned char* labels, int* labelDest) {

	*labelDest = labels[*labelCounter];
	*labelCounter = *labelCounter + 1;
}


void NeuralNetwork::trainNetwork(Mat trainingInput, float trainingLabel) {

	purgeGradientDescentVectors();
	fireNeuralNetwork(trainingInput, trainingLabel);
	calculateHiddenLayerGradients();
	calculateOutputLayerGradients();
	computeDeltaVs();
	UpdateNetworkVariables();

}


void NeuralNetwork::trainNetwork_MNIST(int trainingSetSize, unsigned char* trainingImages, unsigned char* trainingLabels) {

	Mat img(28,28,CV_8UC1);
	int* label = new int(0);

	int* imgCounter = new int(0);
	int* labelCounter = new int(0);

	for (int i = 0; i < trainingSetSize; i++)
	{
		dispense_mnist_image(imgCounter, trainingImages, img);
		dispense_mnist_label(labelCounter, trainingLabels, label);
	
		trainNetwork(img,(float) *label);


		/*
		if(i == 5972 || i == 5973)
		{

			cout << "INPUT LAYER INFO:" << endl << endl << endl;
			for(unsigned int x = 0; x < inputLayer.size();x++)
			{
				cout << "Neuron#" << x << " output=" << inputLayer[x]->output << endl;
			}

			cout << endl << endl << endl << endl;
			cout << "HIDDEN LAYER INFO:" << endl << endl << endl << endl;
			for(unsigned int x = 0; x < hiddenLayer.size();x++)
			{
				cout << "Neuron#" << x << endl;
				cout << "Bias=" << hiddenLayer[x]->bias << endl;
				cout << "Z=" << hiddenLayer[x]->sigmoidInput_z << endl;
				cout << "Output=" << hiddenLayer[x]->output << endl;
				for (int z = 0; z < inputLayer.size();z++)
				{
					cout << "InputWeight=" << hiddenLayer[x]->inputWeights[z] << endl;
				}
			}

			cout << endl << endl << endl << endl;
			cout << "OUTPUT LAYER INFO:" << endl << endl << endl << endl;
	
			for(unsigned int x = 0; x < outputLayer.size(); x++)
			{
				cout << "Neuron#" << x << endl;
				cout << "Bias=" << outputLayer[x]->bias << endl;
				cout << "Z=" << outputLayer[x]->sigmoidInput_z << endl;
				cout << "Output=" << outputLayer[x]->output << endl;
				for (int z = 0; z < hiddenLayer.size(); z++)
				{
					cout << "InputWeight=" << outputLayer[x]->inputWeights[z] << endl;
				}
			}

			namedWindow("test",1);
			imshow("test",img);
			waitKey(0);
		}
		*/



		
		cout << "Training Run: " << i << endl;
		cout << "OutputNeurons Actual--->   ";
		for (unsigned int x = 0; x < outputLayer.size(); x++)
		{
			cout << "[" << outputLayer[x]->output <<"]";
		}

		cout << endl;

		cout << "OutputNeurons Target--->   ";
		for (unsigned int x = 0; x < outputLayer.size(); x++)
		{
			if (x == *label)
			{
				cout << "[1]";
			}
			else
			{
				cout << "[0]";
			}
		}
		cout << endl;
		cout << endl;
		
	}

}


int main(int argc, char** argv)
{
	// Load images and labels into memory
	unsigned char* images = read_mnist_images(argv[1]);
	unsigned char* labels = read_mnist_labels(argv[2]);

	// Get img dimensions for network initialization
	Mat img(28,28,CV_8UC1);
	struct imgDimensions dims = getImgDimensions(img);

	// Initialize network and various network components
	NeuralNetwork nn;
	nn.populateInputLayer(dims);
	nn.populateHiddenLayer(15);
	nn.populateOutputLayer(10);
	nn.setDefaultWeights();
	nn.setLearningRate(0.5);


	nn.trainNetwork_MNIST(60000, images, labels);


	return 0;
}

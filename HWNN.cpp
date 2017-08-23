#include<iostream>
#include<vector>
#include<math.h>
#include<stdlib.h>
#include<cv.h>
#include<highgui.h>
#include<cvwimage.h>
#include<random>
#include<fstream>
#include<iomanip>
#include<map>
#include<cmath>
#include</home/julian/MachineLearning/ImageHandler.hpp>
#include<chrono>
#include<pthread.h>
#include<cstdlib>



using namespace std;
using namespace cv;


#define NUM_THREADS 4


struct imgDimensions {

	int rows;
	int cols;
	int channels;
};


Mat getImage(char* imgPath) {

	Mat img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data)
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

// Feeds all outputs from previous layers neuron's into input of each neuron in current layer
void SigmoidNeuron::collectInputs(vector<SigmoidNeuron*>& previousLayer) {

	for (unsigned int i = 0; i < previousLayer.size(); i++)
	{
  		inputValues.push_back(previousLayer[i]->output);
	}
}



// Uses sigmoid function to calculate output value of neuron
float SigmoidNeuron::calculateOutput() {

	float z = 0;

	// Dot product of vectors of weights and inputs
	for (unsigned int i = 0; i < inputWeights.size(); i++)
	{
		// Dot Product: Gets called many times in training...candidate for easy parallelization
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


	NeuralNetwork(Mat img, int hLayerSize, int oLayerSize, float learningRate, ImageHandler* iHandler);

	ImageHandler* iHandler;
	struct imgDimensions networkInputDimensions;

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
	void updateHiddenLayer();
	static void* parallelUpdateHiddenLayer(void* neuron);

	void updateOutputLayer();
	void purgeHiddenLayer();
	void purgeOutputLayer();
	void populateInputLayer();
	void populateHiddenLayer(int hLayerSize);
	void populateOutputLayer(int oLayerSize);
	void fireNeuralNetwork(Mat inputImage, float targetValue);

	void calculateOutputLayerGradients();
	void calculateHiddenLayerGradients();
	void setLearningRate(float learningRate);
	void computeDeltaVs();
	void UpdateNetworkVariables();
	void purgeGradientDescentVectors();
	void computeBatchAverageGradients (int batchSize);
	void batchGradientDescent(int epochs, int batchSize, int trainingSetSize);
	void testNetwork(int testSetSize);


	void printNetworkVariables();
};

NeuralNetwork::NeuralNetwork(Mat img, int hLayerSize, int oLayerSize, float learningRate, ImageHandler* iHandler) {

	networkInputDimensions = getImgDimensions(img);
	populateInputLayer();
	populateHiddenLayer(hLayerSize);
	populateOutputLayer(oLayerSize);
	setDefaultWeights();
	setLearningRate(learningRate);
	this->iHandler = iHandler;
}


// Populates layers with default weights and Biases (must be called after populating
// all layers (using populateXXXXXX functions)
void NeuralNetwork::setDefaultWeights() {

	// Engine for gaussian normal distribution number generation with mean 0 and variation 1
	std::random_device rd;
	std::mt19937 e2(rd());
	std::normal_distribution<> dist(0, 1);

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		for (unsigned int x = 0; x < inputLayer.size(); x++)
		{
			float tempWeight = dist(e2);
			hiddenLayer[i]->inputWeights.push_back(tempWeight);
		}

		float tempBias = dist(e2);
		hiddenLayer[i]->bias = tempBias;
	}

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		for (unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float tempWeight = dist(e2);
			outputLayer[i]->inputWeights.push_back(tempWeight);
		}

		float tempBias = dist(e2);
		outputLayer[i]->bias = tempBias;
	}
}

void NeuralNetwork::feedInputLayer(Mat img) {

	int rows = networkInputDimensions.rows;
	int cols = networkInputDimensions.cols;

	uchar* p = img.data;
	int currentNeuron = 0;

	for (int i = 0; i < rows; i++)
	{
		p = img.ptr<uchar>(i);

		for (int x = 0; x < cols; x++)
		{
			inputLayer[currentNeuron]->output = (float)p[x] / 255.00;
			currentNeuron++;

		}
	}

}


// Target values start at 0 (e.g if target is 1 then 0th node in outputLayer should be 1, so set target to 0)
void NeuralNetwork::setTargetValue(float targetValue) {

	this->targetValue = targetValue;
}


struct pkg {

	vector<SigmoidNeuron*>* IL;
	vector<SigmoidNeuron*>* HL;
	unsigned int startNeuronIndex;
	unsigned int endNeuronIndex;
};


void *NeuralNetwork::parallelUpdateHiddenLayer(void* input) {

	struct pkg* P = (struct pkg*) input;

	vector<SigmoidNeuron*>* il = P->IL;
	vector<SigmoidNeuron*>* hl = P->HL;

	for(unsigned int i = P->startNeuronIndex; i <= P->endNeuronIndex; i++)
	{
		*hl[i]->collectInputs(*P->IL);
		*hl[i]->calculateOutput();
	}

}


void NeuralNetwork::updateHiddenLayer() {

	pthread_t threads[NUM_THREADS];
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	int rc;
	void* status;

	int workload_per_thread = hiddenLayer.size() / NUM_THREADS;
	int remainder_work = hiddenLayer.size() % NUM_THREADS;

	struct pkg* thread1_workload = new pkg();
	thread1_workload->IL = &inputLayer;
	thread1_workload->HL = &hiddenLayer;
	thread1_workload->startNeuronIndex = 0;
	thread1_workload->endNeuronIndex = workload_per_thread - 1;
	pthread_create(&threads[0], &attr, NeuralNetwork::parallelUpdateHiddenLayer, (void*) thread1_workload);

	struct pkg* thread2_workload = new pkg();
	thread1_workload->IL = &inputLayer;
	thread1_workload->HL = &hiddenLayer;
	thread2_workload->startNeuronIndex = thread1_workload->endNeuronIndex + 1;
	thread2_workload->endNeuronIndex = thread2_workload->startNeuronIndex + workload_per_thread -1;
	pthread_create(&threads[1], &attr, NeuralNetwork::parallelUpdateHiddenLayer, (void*) thread2_workload);

	struct pkg* thread3_workload = new pkg();
	thread1_workload->IL = &inputLayer;
	thread1_workload->HL = &hiddenLayer;
	thread3_workload->startNeuronIndex = thread2_workload->endNeuronIndex + 1;
	thread3_workload->endNeuronIndex = thread3_workload->startNeuronIndex + workload_per_thread - 1;
	pthread_create(&threads[2], &attr, NeuralNetwork::parallelUpdateHiddenLayer, (void*) thread3_workload);

	struct pkg* thread4_workload = new pkg();
	thread1_workload->IL = &inputLayer;
	thread1_workload->HL = &hiddenLayer;
	thread4_workload->startNeuronIndex = thread3_workload->endNeuronIndex + 1;
	thread4_workload->endNeuronIndex = thread4_workload->startNeuronIndex + workload_per_thread - 1 + remainder_work;


	for(int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(threads[i], &status);
	}
}

void NeuralNetwork::purgeHiddenLayer() {

	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayer[i]->inputValues.clear();
	}
}

void NeuralNetwork::updateOutputLayer() {

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i]->collectInputs(hiddenLayer);
		outputLayer[i]->calculateOutput();
	}
}

void NeuralNetwork::purgeOutputLayer() {

	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i]->inputValues.clear();
	}

}

// Populates input layer of network with appropriate number of neurons. Expects 
// that networkInputDimensions has already been set by network constructor.
void NeuralNetwork::populateInputLayer() {

	int rows = networkInputDimensions.rows;
	int cols = networkInputDimensions.cols;

	for (int i = 0; i < rows*cols; i++)
	{
		this->inputLayer.push_back(new SigmoidNeuron);
	}

}

void NeuralNetwork::populateHiddenLayer(int hLayerSize) {

	for (int i = 0; i < hLayerSize; i++)
	{
		this->hiddenLayer.push_back(new SigmoidNeuron);
	}
}

void NeuralNetwork::populateOutputLayer(int oLayerSize) {

	for (int i = 0; i < oLayerSize; i++)
	{
		this->outputLayer.push_back(new SigmoidNeuron);
	}
}

void NeuralNetwork::fireNeuralNetwork(Mat inputImage, float targetValue) {


	purgeHiddenLayer();
	purgeOutputLayer();
	feedInputLayer(inputImage);
	setTargetValue(targetValue);

	// Need to block until updateHiddenLayer workers are finished...
	updateHiddenLayer();
	updateOutputLayer();

}


void NeuralNetwork::calculateOutputLayerGradients() {

	vector<float> targetValues(outputLayer.size()); // t sub k
	vector<float> actualValues(outputLayer.size()); // a sub k
	vector<float> outputLayerInputs_z(outputLayer.size()); // z sub k
	vector<float> hiddenLayerOutputs(hiddenLayer.size()); // a sub j

	// Collecting values 
	for (unsigned int i = 0; i < outputLayer.size(); i++)
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
	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
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
		tempBiasGradient = (AV - TV) * ((exp(-z)) / powf(exp(-z) + 1, 2.00));

		outputLayerBiasGradients.push_back(tempBiasGradient);

		for (unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			float a = hiddenLayerOutputs[x];
			float tempWeightGradient = tempBiasGradient;
			// Output layer weight gradient for weight connecting node x of hidden layer to 
			// node i of output layer
			tempWeightGradient = tempWeightGradient * a;
			outputLayerWeightGradients.push_back(tempWeightGradient);
		}
	}




}

void NeuralNetwork::calculateHiddenLayerGradients() {

	vector<float> inputLayerOutput_NodeI(inputLayer.size()); // a sub i
	vector<float> hiddenLayerInputs_z(hiddenLayer.size()); // z sub j
	vector<float> outputLayerOutputs(outputLayer.size()); // a sub k
	vector<float> targetOutput(outputLayer.size());	// t sub k
	vector<float> outputLayerInputs_z(outputLayer.size()); // z sub k

	// The size of this 2D array has been hardcoded since standard C++ doesn't support
	// dynamic allocation of arrays....need to fix this eventually
	float hiddenLayertoOutputLayerWeights[outputLayer.size()][hiddenLayer.size()]; // w sub jk


	// sets a sub i (correct)
	for (unsigned int i = 0; i < inputLayer.size(); i++)
	{

		inputLayerOutput_NodeI[i] = inputLayer[i]->output;
	}

	// sets z sub j (correct)
	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		hiddenLayerInputs_z[i] = hiddenLayer[i]->sigmoidInput_z;
	}

	// Sets a sub k && t sub k && z sub k (correct)	
	for (unsigned int i = 0; i < outputLayer.size(); i++)
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

	// Sets w sub jk (correct)
	for (unsigned int i = 0; i < outputLayer.size(); i++)
	{
		for (unsigned int x = 0; x < hiddenLayer.size(); x++)
		{
			hiddenLayertoOutputLayerWeights[i][x] = outputLayer[i]->inputWeights[x];
		}
	}



	// Start gradient calculations
	for (unsigned int i = 0; i < hiddenLayer.size(); i++)
	{
		// correct
		int z_temp = hiddenLayerInputs_z[i];
		float term1 = exp(-z_temp);
		float term2 = (1 + term1) * (1 + term1);
		float gPrimeOfJ = term1 / term2;

		float dSubK;
		float Sum = 0;

		for (unsigned int x = 0; x < outputLayer.size(); x++)
		{
			dSubK = (outputLayerOutputs[x] - targetOutput[x]) * (exp(-outputLayerInputs_z[x]) / powf(exp(-outputLayerInputs_z[x]) + 1, 2.00));
			dSubK = dSubK * hiddenLayertoOutputLayerWeights[x][i];
			Sum += dSubK;

		}


		float biasGradientTemp = gPrimeOfJ * Sum;
		this->hiddenLayerBiasGradients.push_back(biasGradientTemp);

		for (unsigned int x = 0; x < inputLayer.size(); x++)
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
		float tempDeltaV = (learningRate)*(outputLayerBiasGradients[i]);
		outputLayerBiasDeltaVs.push_back(tempDeltaV);
	}

	for (unsigned int i = 0; i < outputLayerWeightGradients.size(); i++)
	{
		float tempDeltaV = (learningRate)*(outputLayerWeightGradients[i]);
		outputLayerWeightDeltaVs.push_back(tempDeltaV);
	}

	for (unsigned int i = 0; i < hiddenLayerBiasGradients.size(); i++)
	{
		float tempDeltaV = (learningRate)*(hiddenLayerBiasGradients[i]);
		hiddenLayerBiasDeltaVs.push_back(tempDeltaV);
	}

	for (unsigned int i = 0; i < hiddenLayerWeightGradients.size(); i++)
	{
		float tempDeltaV = (learningRate)*(hiddenLayerWeightGradients[i]);
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
			outputLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron] = (outputLayer[i]->inputWeights[inputWeightIndexForCurrentNeuron]) - (outputLayerWeightDeltaVs[x]);
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


// Computes average gradient for a given batch 
void NeuralNetwork::computeBatchAverageGradients (int batchSize) {

	vector<float> averageHiddenLayerBiasGradients(hiddenLayer.size());
	vector<float> averageHiddenLayerWeightGradients(inputLayer.size() * hiddenLayer.size());
	vector<float> averageOutputLayerBiasGradients(outputLayer.size());
	vector<float> averageOutputLayerWeightGradients(hiddenLayer.size() * outputLayer.size());

	int rows = networkInputDimensions.rows;
	int cols = networkInputDimensions.cols;
	Mat img (rows,cols, CV_8UC1);
	int label;

	for(int i = 0; i < batchSize; i++)
	{
		purgeGradientDescentVectors();
		img = iHandler->dispense_image();
		label = iHandler->dispense_label();
		fireNeuralNetwork(img, (float) label);
		calculateHiddenLayerGradients();
		calculateOutputLayerGradients();

		if(i == 0)
		{
			averageHiddenLayerBiasGradients = hiddenLayerBiasGradients;
			averageHiddenLayerWeightGradients = hiddenLayerWeightGradients;
			averageOutputLayerBiasGradients = outputLayerBiasGradients;
			averageOutputLayerWeightGradients = outputLayerWeightGradients;
		}
		else
		{
			for(unsigned int x = 0; x < averageHiddenLayerBiasGradients.size(); x++)
			{
				averageHiddenLayerBiasGradients[x] += hiddenLayerBiasGradients[x];
			}

			for(unsigned int x = 0; x < averageHiddenLayerWeightGradients.size(); x++)
			{
				averageHiddenLayerWeightGradients[x] += hiddenLayerWeightGradients[x];
			}

			for(unsigned int x = 0; x < averageOutputLayerBiasGradients.size(); x++)
			{
				averageOutputLayerBiasGradients[x] += outputLayerBiasGradients[x];
			}

			for(unsigned int x = 0; x < averageOutputLayerWeightGradients.size(); x++)
			{
				averageOutputLayerWeightGradients[x] += outputLayerWeightGradients[x];
			}
		}
	}



	for(unsigned int x = 0; x < averageHiddenLayerBiasGradients.size(); x++)
	{
		averageHiddenLayerBiasGradients[x] = averageHiddenLayerBiasGradients[x] / batchSize; 
	}

	for(unsigned int x = 0; x < averageHiddenLayerWeightGradients.size(); x++)
	{
		averageHiddenLayerWeightGradients[x] = averageHiddenLayerWeightGradients[x] / batchSize;
	}

	for(unsigned int x = 0; x < averageOutputLayerBiasGradients.size(); x++)
	{
		averageOutputLayerBiasGradients[x] = averageOutputLayerBiasGradients[x] / batchSize;
	}

	for(unsigned int x = 0; x < averageOutputLayerWeightGradients.size(); x++)
	{
		averageOutputLayerWeightGradients[x] = averageOutputLayerWeightGradients[x] / batchSize;
	}


	purgeGradientDescentVectors();

	hiddenLayerBiasGradients = averageHiddenLayerBiasGradients;
	hiddenLayerWeightGradients = averageHiddenLayerWeightGradients;
	outputLayerBiasGradients = averageOutputLayerBiasGradients;
	outputLayerWeightGradients = averageOutputLayerWeightGradients;
}

void NeuralNetwork::batchGradientDescent(int epochs, int batchSize, int trainingSetSize) {

	// Run all epochs
	for(int epoch = 0; epoch < epochs; epoch++)
	{
		// Run all batches in current epoch
		for (unsigned int g = 0; g < trainingSetSize / batchSize; g++)
		{
			computeBatchAverageGradients(batchSize);
			computeDeltaVs();
			UpdateNetworkVariables();
		}
	}			
}

// Prints all weights and biases for each layer in network
void NeuralNetwork::printNetworkVariables() {

	cout << "Hidden Layer Bias':" << endl;
	for (int i = 0; i < hiddenLayer.size(); i++)
	{
		cout << "[" << hiddenLayer[i]->bias << "] ";
	}
	cout << endl << endl;


	cout << "Hidden Layer Weights:" << endl;
	for(int i = 0; i < hiddenLayer.size(); i++)
	{
		cout << "Hidden Layer Neuron=" << i << " Weights:" << endl;
		for (int x = 0; x < hiddenLayer[i]->inputWeights.size(); x++)
		{
			cout << "[" << hiddenLayer[i]->inputWeights[x] << "] ";
		}
		cout << endl << endl;
	}
	cout << endl << endl;

	cout << "Output Layer Bias':" << endl;
	for (int i = 0; i < outputLayer.size(); i++)
	{
		cout << "[" << outputLayer[i]->bias << "] ";
	}
	cout << endl << endl;
	
	cout << "Output Layer Weights:" << endl;
	for(int i = 0; i < outputLayer.size(); i++)
	{
		cout << "Output Layer Neuron=" << i << " Weights:" << endl;
		for(int x = 0; x < outputLayer[i]->inputWeights.size(); x++)
		{
			cout << "[" << outputLayer[i]->inputWeights[x] << "] ";
		}
		cout << endl << endl;
	}
	cout << endl << endl;


}


void NeuralNetwork::testNetwork(int testSetSize) {


	int rows = networkInputDimensions.rows;
	int cols = networkInputDimensions.cols;
	Mat img(rows, cols, CV_8UC1);
	int label;

	int correct = 0;
	int incorrect = 0;

	for(int i = 0; i < testSetSize; i++)
	{
		img = iHandler->dispense_image();
		label = iHandler->dispense_label();
		fireNeuralNetwork(img, (float)label);

		int maxIndex = 0;


		cout << "Output Layer Actual Output (Label=" << label << ")--->   ";
		for (int x = 0; x < outputLayer.size(); x++)
		{
			cout << "[" << outputLayer[x]->output << "]";
			
			if(outputLayer[x]->output > outputLayer[maxIndex]->output)
			{
				maxIndex = x;
			}
		}
		cout << endl << endl;
		if(maxIndex == label)
		{
			correct++;
		}
		else
		{
			incorrect++;
		}
	}	

	cout << endl << endl << endl;
	cout << "Correct=" << correct << endl;
	cout << "Incorrect=" << incorrect << endl;
}






// Set network initialization parameters
Mat img(28, 28, CV_8UC1); // Set matrix dimensions (type must be CV_8UC1 as network cannot handle other types)
int hiddenLayerSize = 32;
int outputLayerSize = 10;
int learningRate = 3;

// Set gradient descent initialization parameters
int trainImageSetSize = 60000;
int batchSize = 20;
int epochs = 1;

// Set number of images in test set
int testImageSetSize = 10000;


int main(int argc, char** argv)
{
	// Train network
	MNISTImageHandler* imageHandler = new MNISTImageHandler(28,28,argv[1],argv[2]);
	NeuralNetwork nn(img, hiddenLayerSize, outputLayerSize, learningRate, imageHandler);
	nn.batchGradientDescent(epochs, batchSize, trainImageSetSize);

	
	// Test network
	MNISTImageHandler* imageHandler2 = new MNISTImageHandler(28,28,argv[3],argv[4]);
	nn.iHandler = imageHandler2;	
	nn.testNetwork(testImageSetSize);
	

	return 0;
}

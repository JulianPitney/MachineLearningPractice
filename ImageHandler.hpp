#include<iostream>
#include<fstream>
#include<cv.h>

using namespace std;
using namespace cv;



class ImageHandler {

public:
	virtual Mat dispense_image(){};
	virtual void load_images(){};
	virtual int dispense_label(){};
	virtual void load_labels(){};
};



class MNISTImageHandler: public ImageHandler {

public:

	Mat dispense_image();
	void load_images();
	int dispense_label();
	void load_labels();


	MNISTImageHandler(int rows, int cols, const char* imagesPath, const char* labelsPath);


	// For loading+dispensing images
	int* imgCounter;
	unsigned char* images;
	int imageSetSize;
	int currentImage;
	const char* MNISTImagesPath;

	// For loading+dispensing labels
	int* labelCounter;
	unsigned char* labels;
	int labelSetSize;
	int currentLabel;
	int* labelDest;
	const char* MNISTLabelsPath;

	
	// General functions
	int reverseInt(int i);
};




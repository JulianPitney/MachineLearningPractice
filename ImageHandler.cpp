#include</home/julian/MachineLearning/ImageHandler.hpp>


MNISTImageHandler::MNISTImageHandler(int rows, int cols, const char* imagesPath, const char* labelsPath) {


	imgCounter = new int(0);
	MNISTImagesPath = imagesPath;

	labelCounter = new int(0);
	labelDest = new int(0);
	MNISTLabelsPath = labelsPath;
	load_images();
	load_labels();
}

int MNISTImageHandler::reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



// Loads MNIST Images found at MNISTImagesPath into images[]
void MNISTImageHandler::load_images()
{
	ifstream file(MNISTImagesPath);
	unsigned char* imagesAddress;

	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = this->reverseInt(magic_number);

		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = this->reverseInt(number_of_images);

		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = this->reverseInt(n_rows);

		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = this->reverseInt(n_cols);

		imagesAddress = new unsigned char[number_of_images * 784];
		int imagesCounter = 0;

		for (int i = 0; i<number_of_images; ++i)
		{
			for (int r = 0; r<n_rows; ++r)
			{
				for (int c = 0; c<n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					imagesAddress[imagesCounter] = temp;
					imagesCounter++;
				}
			}
		}
	}
	
	images = imagesAddress;	
}





// Dispenses image from images[] in the MNISTImageHandler. 
// Uses imgCounter in MNISTImageHandler to keep track of current
// position in images[].
Mat MNISTImageHandler::dispense_image() {

	Mat img(28,28, CV_8UC1);
	uchar* p = img.data;

	for (unsigned int row = 0; row < 28; row++)
	{
		p = img.ptr<uchar>(row);

		for (unsigned int col = 0; col < 28; col++)
		{
			p[col] = images[*imgCounter];
			*imgCounter = *imgCounter + 1;
		}
	}

	return img;
}


// Loads MNIST Labels into labes[] in MNISTImageHandler.
// (Looks for file at MNISTLabelsPath in MNISTImageHandler).
void MNISTImageHandler::load_labels() {

	ifstream file(MNISTLabelsPath);
	unsigned char* labelsAddress;


	if (file.is_open())
	{

		int magic_number = 0;
		int number_of_labels = 0;

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = this->reverseInt(magic_number);

		file.read((char*)&number_of_labels, sizeof(number_of_labels));
		number_of_labels = this->reverseInt(number_of_labels);

		labelsAddress = new unsigned char[number_of_labels];

		for (unsigned int i = 0; i < number_of_labels; i++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			labelsAddress[i] = temp;
		}
	}

	labels = labelsAddress;
}

// Returns value of label contained in labels[] at labelCounter.
int MNISTImageHandler::dispense_label() {

	int label = labels[*labelCounter];
	*labelCounter = *labelCounter + 1;
	return label;
}

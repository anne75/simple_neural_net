#include <armadillo>
#include <math.h>
#include <iostream>


/*set up the parameters*/
//unsigned input_layer_size = 400;
//unsigned hidden_layer_size = 25;
//unsigned num_labels = 10;


/* loading the data */
/* http://eric-yuan.me/cpp-read-mnist/ */
/* https://stackoverflow.com/a/10409376/7484498 */
using namespace std;
using namespace arma;

int ReverseInt (int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, arma::mat &vec)
{
	ifstream file (filename.c_str(), ios::binary);

	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for(int i = 0; i < number_of_images; ++i)
		{
			for(int r = 0; r < n_rows; ++r)
			{
				for(int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));
					vec(i, r * n_rows + c) = (double) temp;
				}
			}
		}
	}
}

void read_Mnist_Label(string filename, arma::mat &vec)
{
	ifstream file (filename.c_str(), ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for(int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec(i, 0)= (double)temp;
		}
	}
}

int test_main()
{
	string filename = "train-images.idx3-ubyte";
	int number_of_images = 60000;
	int image_size = 28 * 28;

	//read MNIST image into Armadillo mat vector
	arma::mat vec(number_of_images, image_size);
	read_Mnist(filename, vec);
	cout << "size of vec " << vec.size() << endl;
	cout << "size of image " << vec.n_cols << endl;
	cout << "first image " << vec.row(0) << endl;

	filename = "train-labels.idx1-ubyte";
    //read MNIST label into armadillo colvec
    //if you want rowvec, just use .t()
	arma::mat vec_label(number_of_images, 1);
	read_Mnist_Label(filename, vec_label);
	cout << "size fo labels " << vec_label.size() << endl;

	return 0;
}

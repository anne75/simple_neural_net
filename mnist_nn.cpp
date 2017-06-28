#include "neural_network.h"

using namespace std;
using namespace arma;


int main() {
	string filename = "train-images.idx3-ubyte";
	int number_of_images = 60000;
	int image_size = 28 * 28;
	nn essai;
	int i, count, count_train, train_sample, test_sample;

	//read MNIST image into Armadillo mat
	arma::mat X(number_of_images, image_size);
	read_Mnist(filename, X);

	filename = "train-labels.idx1-ubyte";
        //read MNIST label into armadillo mat
	arma::mat y(number_of_images, 1);
	read_Mnist_Label(filename, y);

	train_sample = 3000;
	mat trainX = X.head_rows(train_sample);
	mat trainy = y.head_rows(train_sample);
	essai.A = &trainX;
	essai.B = &trainy;
	essai.input_layer_size = image_size;
	essai.hidden_layer_size = 25;
	essai.num_labels = 10;
	essai.lambda = 1;
	essai.max_iter = 50;

	field<mat> data = train_nn(&essai);
	//cout << data.size() << endl;
	//cout << data.n_rows << endl;
	//cout << data.n_cols << endl;
	mat theta1 = data[0];
	mat theta2 = data[1];

	test_sample = 500;
	mat X_test = X.tail_rows(test_sample);
	ucolvec result = predict(theta1, theta2, X_test);
	mat y_test = y.tail_rows(test_sample);
	count = 0;
	for (i = 0; i < test_sample; ++i)
	{
		//cout << y_test(i, 0) << " " << result[i] << endl;
		if (y_test(i, 0) == result[i])
			++count;
	}
	result = predict(theta1, theta2, trainX);
	count_train = 0;
	for (i = 0; i < train_sample; ++i)
	{
		if (trainy(i, 0) == result[i])
			++count_train;
	}
	cout << essai.lambda << "," << essai.max_iter << ","
	     << (count * 1.0) / test_sample << ","
	     << (count_train * 1.0) / train_sample << endl;
	return (0);
}

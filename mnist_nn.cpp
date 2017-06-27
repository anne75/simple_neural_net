#include "neural_network.h"

using namespace std;
using namespace arma;


int main() {
	string filename = "train-images.idx3-ubyte";
	int number_of_images = 60000;
	int image_size = 28 * 28;
	nn essai;

	//read MNIST image into Armadillo mat
	arma::mat X(number_of_images, image_size);
	read_Mnist(filename, X);
	cout << "size of X " << X.size() << endl;
	cout << "size of image " << X.n_cols << endl;

	filename = "train-labels.idx1-ubyte";
        //read MNIST label into armadillo mat
	arma::mat y(number_of_images, 1);
	read_Mnist_Label(filename, y);
	cout << "size fo labels " << y.size() << endl;

	mat trainX = X.head_rows(2000);
	mat trainy = y.head_rows(2000);
	essai.A = &trainX;
	essai.B = &trainy;
	essai.input_layer_size = 784;
	essai.hidden_layer_size = 25;
	essai.num_labels = 10;
	essai.lambda = 2;
	essai.max_iter = 50;

	cout << "train X :" << trainX.n_rows << endl;
	field<mat> data = train_nn(&essai);
	//cout << data.size() << endl;
	//cout << data.n_rows << endl;
	//cout << data.n_cols << endl;
	mat theta1 = data[0];
	mat theta2 = data[1];
	cout << "return from training" << endl;
	cout << theta1.n_rows << endl;
	cout << theta1.n_cols << endl;
	cout << theta2.n_rows << endl;
	cout << theta2.n_cols << endl;

	cout << "predict" << endl;
	mat X_test = X.tail_rows(20);
	ucolvec result = predict(theta1, theta2, X_test);
	mat y_test = y.tail_rows(20);
	double count = 0;
	for (int i = 0; i < 20; ++i)
	{
		cout << y_test(i, 0) << " " << result[i] << endl;
		if (y_test(i, 0) == result[i])
		{
			cout << "here" << endl;
			++count;
		}
	}
	cout << "precision: " << count / 20 << endl;
	return (0);
}

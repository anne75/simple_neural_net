#include "neural_network.h"

using namespace std;
using namespace arma;


/**
 * randomly initialize the weights of a layer with L_in incoming
 * connections and L_out outgoing connections
 */
arma::mat randInitializeWeights(unsigned L_in, unsigned L_out)
{
	double epsilon_init = 0.12;
	arma::mat W = arma::zeros<arma::mat>(L_out, 1 + L_in);
	W = randu(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
	return W;
}

arma::mat sigmoid(mat &z)
{
	return (1.0 / (1.0 + exp(-z)));
}

double costFunction(const std::vector<double> &nn_params,
		    std::vector<double> &grad, void* data)
{
	nn *essai = (nn *)data;
	double J = 0;

	/*remake thetas*/
	colvec params = conv_to< mat >::from(nn_params);

	mat theta1 = params.rows(0, (essai->hidden_layer_size) *
				 ((essai->input_layer_size) + 1));
	theta1.reshape(essai->hidden_layer_size,
		       essai->input_layer_size + 1);
	mat theta2 = params.rows((essai->hidden_layer_size) *
				 ((essai->input_layer_size) + 1),
				 params.n_rows - 1);
	theta2.reshape(essai->num_labels,
		       essai->hidden_layer_size + 1);

	/*extract training set*/
	//mat X = ((mat *)data)[0];
	//mat y = ((mat *)data)[1];
	mat X = *(essai->A);
	mat y = *(essai->B);
	
	unsigned x_rows = X.n_rows;

	/*feed forward*/
	/*prepare X: add the bias, and transpose it to make it like a1*/
	X.insert_cols(0, ones<mat>(x_rows, 1));
	X = X.t();
	/*calculate a2*/
	mat a2 = theta1 * X;
	a2 = sigmoid(a2);
	/*prepare a2*/
	a2.insert_rows(0, ones<mat>(1, a2.n_cols));
	//cout << a2 << "a2 first" << endl;
	/*calculate a3*/
	mat a3 = theta2 * a2;
	a3 = sigmoid(a3);
	/*modify y to make it a matrix of size x_rows * num_labels*/
	mat tmp = repmat(linspace(0, (essai->num_labels) - 1,
				  essai->num_labels).t(), x_rows, 1);
	umat Y = (tmp == repmat(y, 1, essai->num_labels));
	/*now compute the cost J, note that when doing Y * a3 I only need the diagonal*/
	tmp = Y * log(a3);
	mat tmp2 = (1 - Y) * log(1 - a3);
	J = accu(-tmp.diag() - tmp2.diag()) / x_rows;
	/*add regularisation*/
	J += (essai->lambda) / (2.0 * x_rows) *
		(accu(square(theta1.cols(1, theta1.n_cols-1)))
				   + accu(square(theta2.cols(1, theta2.n_cols-1)
						  )));
	cout << "cost: " << J << endl;

	/*backpropagation, I do not use a loop*/
	mat theta1_grad = zeros<mat>(theta1.n_rows, theta1.n_cols);
	mat theta2_grad = zeros<mat>(theta2.n_rows, theta2.n_cols);
	mat delta3 = a3 - Y.t();
	mat delta2 = (theta2.t() * delta3) % (a2 % ( 1 - a2));
	delta2.shed_row(0);
	theta1_grad = theta1_grad + delta2 * X.t();
	theta2_grad = theta2_grad + delta3 * (a2.t());
	//cout << "value" << endl;
	//cout << delta3 << "delta3" << endl;
	//cout << a2 << "a2" << endl;
	//cout << delta3 * a2.t() << endl;
	//cout << theta2_grad << endl;
	/*add regularisation*/
	theta1_grad = (1.0 / x_rows) * theta1_grad +
		((essai->lambda) / x_rows) * theta1;
	theta2_grad = (1.0/x_rows) * theta2_grad +
		((essai->lambda) / x_rows) * theta2;
	//Remove regularization for Bias term gradients
	theta1_grad.col(0) = theta1_grad.col(0) -
		((essai->lambda) / x_rows) * (theta1.col(0));
	theta2_grad.col(0) = theta2_grad.col(0) -
		((essai->lambda) / x_rows) * (theta2.col(0));

	/*unroll the gradients*/
	vector<double> gradients = conv_to< vector<double> >::from(
		join_vert(vectorise(theta1_grad), vectorise(theta2_grad)));

	if (!grad.empty())
	{
		//Set nlopt min function gradient vector equal
		//to the unrolled version of our gradients
		grad = gradients;
	}
	return (J);
}

ucolvec predict(mat &theta1, mat &theta2, mat &test)
{
	test.insert_cols(0, ones<mat>(test.n_rows, 1));

	mat a2 = theta1 * test.t();
	a2 = sigmoid(a2);
	a2.insert_rows(0, ones<mat>(1, a2.n_cols));

	mat a3 = theta2 * a2;
	a3 = sigmoid(a3);
	cout << "raw results " << a3 << endl;
	ucolvec result = index_max(a3.t(), 1);
	return (result);
}

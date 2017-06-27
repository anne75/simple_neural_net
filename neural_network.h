#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <armadillo>
#include <math.h>
#include <iostream>
#include <nlopt.hpp>

struct nn {
	arma::mat *A;
	arma::mat *B;
	int input_layer_size;
	int hidden_layer_size;
	int num_labels;
	double lambda;
	int max_iter;
};

arma::mat randInitializeWeights(unsigned L_in, unsigned L_out);
arma::mat sigmoid(arma::mat &z);
double costFunction(const std::vector<double> &nn_params, std::vector<double> &grad, void* data);
arma::ucolvec predict(arma::mat &theta1, arma::mat &theta2, arma::mat &test);

arma::field< arma::mat > train_nn(nn *essai);

/* read_mnist */
int ReverseInt (int i);
void read_Mnist(std::string filename, arma::mat &vec);
void read_Mnist_Label(std::string filename, arma::mat &vec);

#endif

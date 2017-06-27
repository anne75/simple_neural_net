#include "neural_network.h"

using namespace std;
using namespace arma;


int main()
{
	int i, n1, n2;
	unsigned input_layer_size = INPUT_LAYER_SIZE;
	unsigned hidden_layer_size = HIDDEN_LAYER_SIZE;
	unsigned num_labels = NUM_LABELS; /* 0 included */
	double lambda = LAMBDA;
	unsigned training_data_size = 1000;
	mat X = mat(training_data_size,2);
	mat y = mat(training_data_size, 1);

	for (i = 0; i < training_data_size; ++i)
	{
		n1 = (int)(2.0 * rand() /double(RAND_MAX));
		n2 = (int)(2.0 * rand() / double(RAND_MAX));
		X(i, 0) = n1;
		X(i, 1) = n2;
		y(i, 0) = n1 ^ n2;
	}
	cout << "end data" << endl;
	mat data[2];
	data[0] = X;
	data[1] = y;

/* initialize the weigths */
	/*input layer, hidden layer*/
	mat theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	cout << theta1 << endl;
	/*hidden layer, output layer*/
	mat theta2 = randInitializeWeights(hidden_layer_size, num_labels);

/* unroll parameters, convert to thetas vector<double> to use nlopt */
	vector<double> initial_nn_params =
		conv_to< vector<double> >::from(join_vert(vectorise(theta1),
							  vectorise(theta2)));
	int tt = initial_nn_params.size();

/*initialize optimizer*/
	nlopt::opt opt(nlopt::LD_LBFGS, initial_nn_params.size());
	opt.set_min_objective(costFunction, &data);
	//opt.set_ftol_rel(1e-4);
	opt.set_maxeval(100); // try to mimic max number of iterations
	double minf = 0;
	nlopt::result res = opt.optimize(initial_nn_params, minf);
	//costFunction(initial_nn_params, initial_nn_params, &data);

/*get the optimized parameters*/
	cout<<"Cost: "<<minf<<endl;
	colvec params = conv_to< mat >::from(initial_nn_params);
	theta1 = params.rows(0, hidden_layer_size * (input_layer_size + 1));
	theta1.reshape(hidden_layer_size, input_layer_size + 1);
	cout << "theta1" << endl;
	cout << theta1 << endl;
	theta2 = params.rows(hidden_layer_size * (input_layer_size + 1),
			     params.n_rows - 1);
	theta2.reshape(num_labels, hidden_layer_size + 1);


	//Try a test set
	mat test = mat(5,2);
	test(0,0) = 0.0;
	test(0,1) = 0.0;
	test(1, 0) = 1.0;
	test(1, 1) = 1.0;
	test(2, 0) = 2.0;
	test(2, 1) = 2.0;
	test(3, 0) = 2.0;
	test(2, 1) = 0.0;
	test(2, 0) = 0.0;
	test(2, 1) = 2.0;
	cout << "test" << test << endl;
	//make a prediction

	ucolvec result = predict(theta1, theta2, X);
	cout << "prediction:" << endl;
	double good_answers = 0;
	for (i = 0; i < training_data_size; ++i)
	{
		if (result[i] == y(i, 0))
		{
			cout << "True" << endl;
			good_answers += 1;
		}
	}
	good_answers /= training_data_size;
	cout << good_answers <<  endl;

	return 0;
}

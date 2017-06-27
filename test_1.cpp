#include "neural_network.h"

using namespace std;
using namespace arma;


int main()
{
	unsigned input_layer_size = INPUT_LAYER_SIZE;
	unsigned hidden_layer_size = HIDDEN_LAYER_SIZE;
	unsigned num_labels = NUM_LABELS; /* 0 included */
	double lambda = LAMBDA;

	//Training Data - 6 examples
	mat X = mat(6,2);

	X(0,0) = 1.0;
	X(0,1) = 1.0;
	X(1,0) = 3.0;
	X(1,1) = 4.0;
	X(2,0) = 10.0;
	X(2,1) = 10.0;
	X(3,0) = 12.0;
	X(3,1) = 11.0;
	X(4,0) = 2.0;
	X(4,1) = 3.0;
	X(5,0) = 4.0;
	X(5, 1) = 2.0;

	cout << X << endl;
	mat y = mat(6,1);
	y(0,0) = 0.0;
	y(1,0) = 0.0;
	y(2,0) = 1.0;
	y(3,0) = 1.0;
	y(4,0) = 0.0;
	y(5,0) = 0.0;
	cout << y << endl;
	cout << "end data" << endl;
	mat data[2];
	data[0] = X;
	data[1] = y;

/* initialize the weigths */
	mat theta1 = randInitializeWeights(input_layer_size, hidden_layer_size); /*input layer, hidden layer*/
	cout << theta1 << endl;
	mat theta2 = randInitializeWeights(hidden_layer_size, num_labels); /*hidden layer, output layer*/

/* unroll parameters, convert to thetas vector<double> to use nlopt */
	vector<double> initial_nn_params = conv_to< vector<double> >::from(join_vert(vectorise(theta1), vectorise(theta2)));
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
	theta2 = params.rows(hidden_layer_size * (input_layer_size + 1), params.n_rows - 1);
	theta2.reshape(num_labels, hidden_layer_size + 1);


	//Try a test set
	mat test = mat(3,2);
	test(0,0) = 11.0;
	test(0,1) = 11.0;
	test(1, 0) = 3.0;
	test(1, 1) = 1.0;
	test(2, 0) = 2.0;
	test(2, 1) = 4.0;
	cout << "test" << test << endl;
	test.insert_cols(0, ones<mat>(test.n_rows, 1));
	cout << "test" << test << endl;
	//make a prediction

	ucolvec result = predict(theta1, theta2, test);
	cout << "prediction:" << endl;
	cout << result << endl;

	return 0;
}

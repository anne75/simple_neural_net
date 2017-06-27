#include "neural_network.h"

using namespace std;
using namespace arma;


field<mat> train_nn(nn *essai)
{
	field<mat> F(2);

	/* initialize the weigths */
        /*input layer, hidden layer*/
	mat theta1 = randInitializeWeights(
		essai->input_layer_size, essai->hidden_layer_size);
	/*hidden layer, output layer*/
	mat theta2 = randInitializeWeights(
		essai->hidden_layer_size, essai->num_labels);

        /* unroll parameters, convert to thetas vector<double> to use nlopt */
	vector<double> initial_nn_params = conv_to< vector<double> >::from(
		join_vert(vectorise(theta1), vectorise(theta2)));
	int tt = initial_nn_params.size();

        /*initialize optimizer*/
	nlopt::opt opt(nlopt::LD_LBFGS, initial_nn_params.size());
	opt.set_min_objective(costFunction, essai);
	opt.set_maxeval(essai->max_iter);
	double minf = 0;
	nlopt::result res = opt.optimize(initial_nn_params, minf);

        /*get the optimized parameters*/
	cout<<"Cost after optimization: "<< minf << endl;
	colvec params = conv_to< mat >::from(initial_nn_params);
	theta1 = params.rows(0, (essai->hidden_layer_size) *
			     ((essai->input_layer_size) + 1));
	theta1.reshape((essai->hidden_layer_size),
		       (essai->input_layer_size) + 1);
	cout << "optimized theta1 " << theta1.n_rows << endl;
	theta2 = params.rows((essai->hidden_layer_size) *
			     ((essai->input_layer_size) + 1),
			     params.n_rows - 1);
	theta2.reshape(essai->num_labels, essai->hidden_layer_size + 1);

	F[0] = theta1;
	F[1] = theta2;
	return (F);
}

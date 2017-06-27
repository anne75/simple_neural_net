function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%/home/anne/workspace/octave/mlclass-ex4-005/mlclass-ex4


%1ere partie

%je recupere X, j'ajoute des 1 et je transpose pour avoir des colonnes d'observations
X=[ ones(m,1) X];
X=X';
%je calcule a2
a2=sigmoid(Theta1*X);
%je calcule a3
%je rajoute une ligne de 1 a a2
a2=[ones(size(a2,2),1)'; a2];
a3=sigmoid(Theta2*a2);
%je transforme y en une matrice Y de taille m*num_labels
Y=repmat(1:num_labels,m,1);
Y=(Y==repmat(y,1,num_labels));
%pour Y vu par les pros
%subs = [[1:m]', y];
%Y = accumarray(subs, 1, [m,num_labels]);
%je peux calculer J, attention quand je calcule Y*a3 je n'ai besoin que des termes en diagonale
J=sum(-diag(Y*log(a3))-diag((1-Y)*log(1-a3)))/m;
%je rajoute la regularisation
J=J+(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))*lambda/(2*m);

%2eme partie
%je recupere X, j'ajoute des 1 et je transpose pour avoir des colonnes d'observations

%je transforme y en une matrice Y de taille m*num_labels

%j'initialise Delta1 et Delta2
Delta1=zeros(hidden_layer_size,input_layer_size+1);
Delta2=zeros(num_labels,hidden_layer_size+1);
%je boucle

for t=1:m
  a_1=X(:,t);
  z_2=Theta1*a_1;
  a_2=[1;sigmoid(z_2)];
  a_3=sigmoid(Theta2*a_2);
  delta3=(a_3-Y(t,:)');
  delta2=(Theta2(:,2:end)'*delta3).*sigmoidGradient(z_2);
  Delta2=Delta2+delta3*a_2';
  Delta1=Delta1+delta2*a_1'; %je garde TOUS les termes de ai
end

Theta1_grad=Delta1/m;
Theta2_grad=Delta2/m;

%je regularise, j'ajoute des termes uniquement pour les colonnes>=2
temp=Theta1;
temp(:,1)=0;
Theta1_grad=Theta1_grad+lambda*temp/m;
temp=Theta2;
temp(:,1)=0;
Theta2_grad=Theta2_grad+lambda*temp/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

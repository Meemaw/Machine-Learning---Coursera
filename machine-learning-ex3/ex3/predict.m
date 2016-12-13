function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% =========================================================================

a_1 = X; % input layer activation
z_2 = a_1 * Theta1'; % second layer values
a_2 = sigmoid(z_2); % second layer activation
% Add ones to the a_2
a_2 = [ones(size(a_2, 1), 1) a_2];
z_3 = a_2 * Theta2'; % output layer values
h = sigmoid(z_3); % output / hiphothesis

[~, p] = max(h, [], 2);

% =========================================================================


end

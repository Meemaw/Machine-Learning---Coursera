function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% =============================================================

z = X * theta;
h = sigmoid(z); % hipothesis
J_noReg = (1 / m) * sum(-y' * log(h) - (1-y)' * log(1 - h));
grad_noReg = (1 / m) * (h - y)' * X;

% ignore theta(1) for regularization
thetaCorrection = theta;
thetaCorrection(1) = 0;

% compute regularized cost
correction = sum(thetaCorrection .^ 2) * (lambda / (2 * m)); % penalize
J = J_noReg + correction; % cost

% compute regularized gradient
grad = grad_noReg + thetaCorrection' * (lambda / m);

% =============================================================

% return column vector
grad = grad(:);

end

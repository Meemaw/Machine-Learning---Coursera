function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% =============================================================

[J_noReg, grad_noReg] = costFunction(theta, X, y); % use function without regularization

% ignore theta(1) for regularization
thetaCorrection = theta;
thetaCorrection(1) = 0;
correction = sum(thetaCorrection .^ 2) * (lambda / (2 * m)); % regularization
J = J_noReg + correction; % cost


grad = grad_noReg + theta' * (lambda / m); % regularized gradient
grad(1) = grad_noReg(1); % dont penalize theta0


% =============================================================

end

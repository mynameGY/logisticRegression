function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%======Cost Function======
num = 0;
for i=1:m
  h = sigmoid(X(i,:)*theta);
  num = num +(-y(i,1)*log(h) - (1-y(i,1))*log(1-h));
  
end

thetaNum = 0;
for g = 1:length(grad)
  thetaNum = thetaNum + theta(g,1)^2;
end  
J = num/m + (lambda/(2*m))*thetaNum;

%======gradient=======
for g = 1:length(grad)
  gradNum = 0;
  for i=1:m
    h = sigmoid(X(i,:)*theta);
    gradNum = gradNum + (h-y(i,1))*X(i,g)
  end 
  grad(g,1) = gradNum/m + (lambda/m)*theta(g,1);
end    


% =============================================================

end

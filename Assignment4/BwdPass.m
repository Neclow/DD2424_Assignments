function grads = BwdPass(X, Y, RNN, a, h, p)
tau = size(X, 2);
G = -(Y-p)';

% t = tau
htau = h(:,tau+1);   % h_{tau}
h2tau = h(:, tau);   % h_{tau-1}
atau = a(:,tau);
xtau = X(:,tau);
gtau = G(tau,:);     % grad_o

grads.c = gtau';
grads.V = gtau'*htau';

grad_h = gtau*RNN.V;
grad_a = grad_h*diag(1-tanh(atau).^2);

grads.b = grad_a';
grads.W = grad_a'*h2tau';
grads.U = grad_a'*xtau';

for i = flip(1:tau-1)
    gi = G(i,:);
    hi = h(:,i+1);
    h2i = h(:,i);
    ai = a(:,i);
    xi = X(:,i);
    
    grads.c = grads.c + gi';    
    grads.V = grads.V + gi'*hi';
    
    grad_h = gi*RNN.V + grad_a*RNN.W;
    grad_a = grad_h*diag(1-tanh(ai).^2);
    
    grads.b = grads.b + grad_a';
    grads.W = grads.W + grad_a'*h2i';
    grads.U = grads.U + grad_a'*xi';
end
end
function [a, h, p] = FwdPass2(RNN, h0, x0)
% h0: hidden state a time 0, mx1
% x0: first dummy input vector of RNN, size dx1
a = RNN.W*h0 + RNN.U*x0 + RNN.b;
h = tanh(a);
o = RNN.V*h + RNN.c;
p = softmax(o);
end
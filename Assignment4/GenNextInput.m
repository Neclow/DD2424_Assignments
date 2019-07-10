function [Y, gen_txt] = GenNextInput(RNN, h0, x0, n)
%xnext = zeros(RNN.K, 1);
Y = zeros(RNN.K, n);
gen_txt = '';
for t = 1:n
    p = SynthSeq(RNN, h0, x0);
    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    xnext = zeros(RNN.K, 1);
    xnext(ii) = 1;
    Y(:,t) = xnext;
    gen_txt = strcat(gen_txt, ind_to_char(ii));
end
end

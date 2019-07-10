function new_eta = UpdateLearningRate(eta, t, n_s, eta_min, eta_max)
L = 0;
new_eta = eta;
deta = eta_max-eta_min;
while eta == new_eta
    if t >= (2*L*n_s) && t <= (n_s*(2*L+1))
        new_eta = eta_min + deta*(t-2*L*n_s)/n_s;
    elseif t >= (n_s*(2*L+1)) && t <= 2*(L+1)*n_s
        new_eta = eta_max - deta*(t-(2*L+1)*n_s)/n_s;
    end
    L = L + 1;
end
end
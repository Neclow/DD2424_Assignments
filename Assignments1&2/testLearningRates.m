function testLearningRates(t_tot, n_s, eta_min, eta_max)
learningRates = zeros(1,t_tot);

eta = 0.01;

learningRates(1) = eta;

for t = 2:t_tot    
    eta = updateLearningRate(eta, t, n_s, eta_min, eta_max);
    disp(eta);
    learningRates(t) = eta;
end

figure
plot(1:t_tot,learningRates)
grid on

end
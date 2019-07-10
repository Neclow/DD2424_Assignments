function lambda_data = SearchLambda(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b)
nb_tries = 15;
l_min = -5;
l_max = 1;
dl = l_max-l_min;

lambdas = zeros(nb_tries,1);
max_accs = zeros(nb_tries,1);

for i = 1:nb_tries
    disp(['Try ', num2str(i), '/', num2str(nb_tries)]);
    l = l_min + dl*rand(1,1);    
    lambdas(i) = 10^l;
    %[~, ~, ~, ~, ~, ~, ~, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambdas(i));    
    [~, ~, ~, ~, ~, ~, ~, acc_valid] = train(X_train, Y_train, y_train, X_valid, Y_valid, y_valid, GDparams, W, b, lambdas(i));
    max_accs(i) = max(acc_valid);
    disp(['lambda = ', num2str(lambdas(i)), ', acc_valid = ', num2str(max_accs(i))]);
end

% Sorting data (for LaTeX tables)
lambda_data = [lambdas, max_accs];
[~, idx] = sort(lambda_data(:,2), 'descend');
lambda_data = round(lambda_data(idx,:), 4, 'significant');

end
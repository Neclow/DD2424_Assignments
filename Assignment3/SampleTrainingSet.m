function [n, X_s, Y_s, y_s] = SampleTrainingSet(X_train, Y_train, y_train)
K = size(Y_train, 1);
n = zeros(1, K);

for i = 1:size(X_train,2)
    [~, y] = max(Y_train(:,i));
    n(y) = n(y) + 1;
end
n_min = min(n);

inds = zeros(1, K*length(n));

inds(1:n_min) = randsample(1:n(1), n_min);

n_start = n(1);

for i = 1:length(n)-1
    n_end = n_start + n(i+1);
    inds(1+n_min*i:n_min*(i+1)) = randsample(n_start+1:n_end, n_min);
    n_start = n_end;
end

X_s = X_train(:,inds);
Y_s = Y_train(:,inds);
y_s = y_train(inds,:);
end
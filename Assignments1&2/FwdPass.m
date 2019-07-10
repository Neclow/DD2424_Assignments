function [P, H, S] = FwdPass(X, W, b)
%Each column of X corresponds to an image, size d x n
%W, size K x d
%P contains proba for each label for the image in the corresponding X
%size K x n

S = cell(numel(b), 1);
H = cell(numel(b) - 1, 1);

S{1} = W{1}*X + b{1};
H{1} = max(0,S{1});

for i = 2:numel(S)
    S{i} = W{i}*H{i-1} + b{i};
    H{i} = max(0,S{i});
end

P = softmax(S{end});
end
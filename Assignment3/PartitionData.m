function Xp = PartitionData(X, inds)

Xp = zeros(size(X,1), length(inds));

for i = 1:length(inds)
    Xp(:,i) = X(:,inds(i));
end
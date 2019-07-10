function grads = ClipGradients(grads)
for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end
end
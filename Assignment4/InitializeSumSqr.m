function m = InitializeSumSqr(RNN)
for f = fieldnames(RNN)'
    m.(f{1}) =  zeros(size(RNN.(f{1})));
end
end
function errs = ComputeRelativeError(ngrads, agrads)
for f = fieldnames(agrads)'
    errs.(f{1}) = ComputeRelativeErrorIndiv(ngrads.(f{1}), agrads.(f{1}));
end
end

function err = ComputeRelativeErrorIndiv(ngrad, agrad)
err = norm(ngrad - agrad)/max(eps, norm(ngrad) + norm(agrad));
end
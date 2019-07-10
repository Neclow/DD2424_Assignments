function err = ComputeRelativeError(ngrad, agrad)
err = norm(ngrad - agrad)/max(eps, norm(ngrad) + norm(agrad));
end
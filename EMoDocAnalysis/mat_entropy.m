function [uncertainty] = mat_entropy(A,dim,inf_val)

if nargin < 2,
    % Default if missing dim argument
    dim = 1;
end
if nargin < 3,
    % Default if missing dim argument
    inf_val = 0;
end

Asum = sum(A,dim);

if issparse(A),
    % Don't want to store whole repmat if A is sparse
    [ii,jj,vv] = find(A);
    if dim == 1,
        vv_norm = vv./Asum(jj)';
    elseif dim == 2,
        vv_norm = vv./Asum(ii);
    else
        % Problem
        return;
    end
    vv_norm(isnan(vv_norm)) = inf_val;
    tmpLog = log10(vv_norm);
    tmpLog(isinf(tmpLog)) = 0;
    entmat = sparse(ii,jj,-1.*vv_norm.*tmpLog);
    uncertainty = sum(entmat,dim);
else
    if dim == 1,
        Anorm = A ./ repmat(Asum,[size(A,dim) 1]); 
    elseif dim == 2,
        Anorm = A ./ repmat(Asum,[1 size(A,dim)]); 
    else
        % Problem
        return;
    end
    Anorm(isnan(Anorm)) = inf_val;
    tmpLog = log10(Anorm);
    tmpLog(isinf(tmpLog)) = 0;
    uncertainty = sum( -1.*Anorm.*tmpLog ,dim);

end


end
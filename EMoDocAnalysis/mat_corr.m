function XX = mat_corr(tdm_norm)
% Fast correlation matrix for sparse matrix input

tic;

tdm_colmean = mean(tdm_norm,1);
[ii,jj,vv] = find(tdm_norm);
vv = vv-tdm_colmean(jj)';
tdm_norm_sub = sparse(ii,jj,vv);

clear('ii','jj','vv');
% NOTE: This line incorrectly tdm_norm in most versions!!!
tdm_colsqrtsumofsq = sqrt(sum(tdm_norm_sub.^2,1));

fprintf(1,'Calculating cov matrix :: '); toc;
XXcov = tdm_norm_sub'*tdm_norm_sub;

fprintf(1,'Calculating product of standard deviations :: '); toc;
XXstdprod = tdm_colsqrtsumofsq'*tdm_colsqrtsumofsq;

fprintf(1,'Calculating correlation matrix :: '); toc;
XX = XXcov./XXstdprod;

% Getting a bunch of NaNs in XX...
XX(isnan(XX)) = 0;

end
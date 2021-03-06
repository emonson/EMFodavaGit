% 20 newsgroups data
% http://people.csail.mit.edu/jrennie/20Newsgroups/
% This version is for data that I re-tokenized with nltk

clear all;

data_dir = '/Users/emonson/Data/Fodava/EMoDocMatlabData';
data_name = 'n20_set2_tdm_121009_combo.mat';
cd(data_dir);
tic;

corrQuantile = 0.90;    % Global cutoff for correlations
NN = 10;                 % number of nn required in corr mtx 
                        % -- includes one self-neighbor

fprintf(1,'Reading training data file\n');
load(data_name);

save_dir = '/Users/emonson/Data/Fodava/EMoDocMatlabData';
% save_dir = '/Volumes/SciVis_LargeData/Fodava';
save_name = 'n20_sub2combo_TFcorr_nn10_021710.mat';

classes = labels';

% calculate Term Frequency (TF) normalization for word counts
nkj = sum(tdm,1)';      % how many terms in each document

[ii,jj,vv] = find(tdm);
vv_norm = vv./nkj(jj);

tdm_norm = sparse(ii,jj,vv_norm);

clear('ii','jj','vv','nkj','D','df','idf','vv_norm');

XX = mat_corr(tdm_norm);

fprintf(1,'Calculating %f quantile of correlation values :: ',corrQuantile); toc;
qq = quantile(XX(XX>0),corrQuantile);

fprintf(1,'Filtering out low corr values :: '); toc;
YY = sparse(XX.*(XX>qq));

% Check if any rows/columns of YY are too sparse
fprintf(1,'Adjusting neighbors :: '); toc;
for ii = find(sum(YY>0,1) < NN),
    % Add elements from XX back into YY to reach required NN count
    [sC,sI] = sort(XX(:,ii),'descend');
    YY(sI(1:NN),ii) = sC(1:NN);
    YY(ii,sI(1:NN)) = sC(1:NN);
end;

numConnComp = graphconncomp(YY);
fprintf(1,'Number of connected components = %d\n', numConnComp);

% For now, break if graph not completely connected...
if (numConnComp > 1)
    break;
end;

clear('XX');

G.W = YY;

clear('YY');

% Compute operators on the graph
fprintf(1,'Constructing graph operators :: '); toc;
G = ConstructGraphOperators ( G );

% Compute the eigenvalues and eigenfunctions
fprintf(1,'Computing eigenvectors/values :: '); toc;
[G.EigenVecs,G.EigenVals] = eigs(G.T,min([100,size(G.W,1)]),'LM',struct('verbose',0));
G.EigenVals = diag(G.EigenVals);

% Compute the multiscale analysis
fprintf(1,'Performing diffusion wavelet analysis :: '); toc;
% WaveletOpts = struct('Symm',true,'Wavelets',false,'GS','gsqr_emo','GSOptions',struct('StopN',4800));
WaveletOpts = struct('Symm',true,'Wavelets',false,'GSOptions',struct('StopN',4800));
G.Tree = DWPTree(G.T, 20, 1e-5, WaveletOpts);

fprintf(1,'Saving file :: '); toc;
cd(save_dir);

% Very large datasets need new version 7.3 save format, but scipy.io can't
% read them...
if (size(G.T,1) > 10000)
    save('-v7.3',save_name);
else
    save(save_name);
end;

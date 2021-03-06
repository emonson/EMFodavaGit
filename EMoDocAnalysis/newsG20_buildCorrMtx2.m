% 20 newsgroups data
% http://people.csail.mit.edu/jrennie/20Newsgroups/

% Original matlab 20news data, but only filtering out <3 on word freq
% and using the faster algorithms for corr and diff wavelets

clear all;
cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DataSets/20news-bydate_matlab/matlab');

corrQuantile = 0.90;    % Global cutoff for correlations
NN = 5;                 % number of nn required in corr mtx 
                        % -- includes one self-neighbor

% read in space-delimited file: docIdx wordIdx count
fprintf(1,'Reading training data file\n');
data = dlmread('train.data');
N = sum(data(:,3));

% Not really normalized data, just using name for convenience...
tdm_norm = sparse(data(:,1),data(:,2),data(:,3))';
clear('data');

classes = dlmread('train.label');
fid = fopen('train.map');
newsgroups = textscan(fid,'%s%d');
fclose(fid);

fprintf(1,'Removing words with frequency < 3\n'); 
wordFreq = sum( tdm_norm>0, 2 );
tdm_norm(wordFreq<3,:) = [];

fprintf(1,'Calculating correlations\n');
tdm_colmean = mean(tdm_norm,1);
[ii,jj,vv] = find(tdm_norm);
vv = vv-tdm_colmean(jj)';
tdm_norm_sub = sparse(ii,jj,vv);

clear('ii','jj','vv');
tdm_colsqrtsumofsq = sqrt(sum(tdm_norm.^2,1));

fprintf(1,'Calculating cov matrix\n');
XXcov = tdm_norm_sub'*tdm_norm_sub;

fprintf(1,'Calculating product of standard deviations\n');
XXstdprod = tdm_colsqrtsumofsq'*tdm_colsqrtsumofsq;

fprintf(1,'Calculating correlation matrix\n');
XX = XXcov./XXstdprod;

% Getting a bunch of NaNs in XX...
XX(isnan(XX)) = 0;

qq = quantile(XX(:),corrQuantile);

YY = sparse(XX.*(XX>qq));

% Check if any rows/columns of YY are too sparse
fprintf(1,'Adjusting neighbors\n');
for ii = find(sum(YY>0,1) < NN),
    % Add elements from XX back into YY to reach required NN count
    [sC,sI] = sort(XX(:,ii),'descend');
    YY(sI(1:NN),ii) = sC(1:NN);
    YY(ii,sI(1:NN)) = sC(1:NN);
end;

G.W = YY;
clear('XX','YY');

% Compute operators on the graph
G = ConstructGraphOperators ( G );

% Compute the eigenvalues and eigenfunctions
% [G.EigenVecs,G.EigenVals] = eigs(G.T,min([100,size(G.W,1)]),'LM',struct('verbose',0));
% G.EigenVals = diag(G.EigenVals);


% Compute the multiscale analysis
WaveletOpts = struct('Symm',true,'Wavelets',false,'GS','gsqr_emo','GSOptions',struct('StopN',4800));
G.Tree = DWPTree(G.T, 20, 1e-5, WaveletOpts);

figure; 
plot(sum(G.Tree{1,1}.ExtBasis,1));
hold on;
plot(sum(G.Tree{2,1}.ExtBasis,1),'r');
plot(sum(G.Tree{3,1}.ExtBasis,1),'g');
plot(sum(G.Tree{4,1}.ExtBasis,1),'m');
plot(sum(G.Tree{5,1}.ExtBasis,1),'k');
plot(sum(G.Tree{6,1}.ExtBasis,1),'c');

figure; 
plot(diag(G.Tree{1,1}.T{1}));
hold on;
plot(diag(G.Tree{2,1}.T{1}),'r');
plot(diag(G.Tree{3,1}.T{1}),'g');
plot(diag(G.Tree{4,1}.T{1}),'m');

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DataSets/20news-bydate_matlab/matlab/LargeData');
save('-v7.3','n20_110509_orig_WF3.mat');


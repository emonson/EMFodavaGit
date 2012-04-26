clear all;
cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
load('X20.mat');

% Get rid of any class = 0 documents
X(classes(:,1)==0,:) = [];
classes(classes(:,1)==0,:) = [];

[ind1,ind2,vv] = find(X);
N = sum(vv);

xr = sum(X,2);
xc = sum(X,1);

Xn = sparse(ind1, ind2, vv./N);
xnr = sum(Xn,2);
xnc = sum(Xn,1);
clear('Xn');

mt = X;
mNNZ = size(vv,1);

fprintf(1,'Calculating mutual information of words in docs\n');
for kk = 1:mNNZ;
    if(~mod(kk,100000)), fprintf(1,'\t%d / %d\n', kk, mNNZ); end;
    ii = ind1(kk);
    jj = ind2(kk);
    xn = vv(kk)./N;
    
    m = log10( xn./(xnr(ii).*xnc(jj)) );    
    c = vv(kk);
    mc = min(xc(jj),xr(ii));
    mt(ii,jj) = m.*(c/(1+c)).*(mc/(1+mc));
end;

clear('ind1','ind2','vv','ii','jj','xc','xr','xnc','xnr');

% --- 
% Create sparse graph with MakeDiffusion2 and knn autotune
% ---
diffOpts = struct('kNN',25,'kNNAutotune',15,'Normalization','sbeltrami','Symmetrization','W+Wt');
% [T,G.W,P,DI] = MakeDiffusion2(full(mt'), 0, diffOpts);
% [T,G.W,P,DI] = MakeDiffusion2(full(X'), 0, diffOpts);

% --- 
% Create graph with MakeDiffusion2 straight knn
% ---
% [T,W,P,DI] = MakeDiffusion2(X',0,struct('kNN',200));
% G.W = T;

% --- 
% Create sparse graph with correlations & cutoff
% ---
corrQuantile = 0.90;    % Global cutoff for correlations
NN = 5;                 % number of nn required in corr mtx (includes one self-neighbor)

fprintf(1,'Calculating correlations of mutual information\n');
XX = corr(mt');
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


% ---
% Compute operators on the graph
% ---
% G.P not good out of MakeDiffusion2 for some reason...
G = ConstructGraphOperators(G);

% Compute the eigenvalues and eigenfunctions
[G.EigenVecs,G.EigenVals] = eigs(G.T,min([100,size(G.W,1)]),'LM',struct('verbose',0));
G.EigenVals = diag(G.EigenVals);


% Compute the multiscale analysis
WaveletOpts = struct('Symm',true,'Wavelets',false); % ,'GS','gsqr_qr');
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
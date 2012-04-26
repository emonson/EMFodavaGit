% 20 newsgroups data
% http://people.csail.mit.edu/jrennie/20Newsgroups/

clear all;
cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DataSets/20news-bydate/matlab');

corrQuantile = 0.90;    % Global cutoff for correlations
NN = 5;                 % number of nn required in corr mtx 
                        % -- includes one self-neighbor

% read in space-delimited file: docIdx wordIdx count
fprintf(1,'Reading training data file\n');
data = dlmread('train.data');
N = sum(data(:,3));

X = sparse(data(:,1),data(:,2),data(:,3));

fprintf(1,'Removing words with frequency < 100\n'); % 96% of words removed
Xs = X;
wordFreq = sum( X>0, 1 );
Xs(:, wordFreq<100) = [];

fprintf(1,'Calculating correlations\n');
XX = corr(Xs');
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

% Compute operators on the graph
G = ConstructGraphOperators ( G );

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

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DataSets/20news-bydate/matlab/LargeData');
save('n20_061909');


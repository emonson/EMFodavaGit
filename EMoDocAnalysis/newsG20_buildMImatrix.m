% 20 newsgroups data
% http://people.csail.mit.edu/jrennie/20Newsgroups/

clear all;
cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DataSets/20news-bydate/matlab');

% read in space-delimited file: docIdx wordIdx count
fprintf(1,'Reading training data file\n');
data = dlmread('train.data');
N = sum(data(:,3));

X = sparse(data(:,1),data(:,2),data(:,3));
xr = sum(X,2);
xc = sum(X,1);
clear('X');

Xn = sparse(data(:,1), data(:,2), data(:,3)./N);
xnr = sum(Xn,2);
xnc = sum(Xn,1);
clear('Xn');

% m = sparse(data(:,1),data(:,2),data(:,3));
mt = sparse(data(:,1),data(:,2),data(:,3));
mNNZ = size(data,1);

fprintf(1,'Calculating mutual information of words in docs\n');
for kk = 1:mNNZ;
    if(~mod(kk,100000)), fprintf(1,'\t%d / %d\n', kk, mNNZ); end;
    ii = data(kk,1);
    jj = data(kk,2);
    xn = data(kk,3)./N;
    
    % m(ii,jj) = log10( xn./(xnr(ii).*xnc(jj)) );
    
    % Current useful for log10
    % m = log10( xn./(xnr(ii).*xnc(jj)) ); 
    
    % Trying low power to avoid negative values, but be close to 
    %   log transformation
    % m = ( xn./(xnr(ii).*xnc(jj)) ).^0.05;    
    m = log10( xn./(xnr(ii).*xnc(jj)) );    
    c = data(kk,3);
    mc = min(xc(jj),xr(ii));
    % mt(ii,jj) = m(ii,jj).*(c/(1+c)).*(mc/(1+mc));
    mt(ii,jj) = m.*(c/(1+c)).*(mc/(1+mc));
end;

clear('data','ii','jj','xc','xr','xnc','xnr');

fprintf(1,'Removing words with frequency < 100\n'); % 96% of words removed
mts = mt;
wordFreq = sum( mt>0, 1 );
mts(:, wordFreq<100) = [];

return;

diffOpts = struct('kNN',25,'kNNAutotune',15,'Normalization','sbeltrami','Symmetrization','W+Wt');
[T,W,P,DI] = MakeDiffusion2(full(mts'),0, diffOpts);


% % taking subset of documents for memory reasons (TESTING)
% docSub = sort(randsample(size(X,1),1000));
% mts = X(docSub,:);
% wordFreq = sum( X>0, 1 );
% mts(:, wordFreq<100) = [];

% clear('mt');
% 
% fprintf(1,'Creating AA\n');
% AA = mts*mts';
% D = sum(AA,2);
% % clear('AA');
% 
% Dinv = diag(D.^-0.5);
% 
% fprintf(1,'mts*D^-0.5\n');
% mtsR = mts'*Dinv;
% 
% fprintf(1,'D^-0.5*mts\n');
% mtsL = Dinv*mts;
% 
% clear('mts');
% 
% fprintf(1,'Creating T = Dinv*AA*Dinv\n');
% T = mtsL*mtsR;
% 
% clear('mtsL','mtsR','Dinv');

WaveletOpts = struct('Symm',true,'Wavelets',false);
G.Tree = DWPTree(T, 20, 1e-5, WaveletOpts);
save('News20Test_061809');


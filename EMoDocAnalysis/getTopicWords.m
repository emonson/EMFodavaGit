% clear all;
% baseName = 'X20_042709b';
% cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
% fprintf(1,'Loading data set\n');
% load([baseName '.mat']);
% load X20.mat
% load ws_dict.mat
% 
% X(classes(:,1)==0,:) = [];
% class = classes;
% class(:,2) = [];
% class(classes(:,1)==0) = [];

% Sci News TF nltk tokenize
clear all;
cd('/Users/emonson/Data/Fodava/EMoDocMatlabData');
fprintf(1,'Loading data set\n');
load('SN_emo1nvj_TFdiff2515_121509.mat');
X = tdm_norm';     % Originally set up for X20.mat which had transposed tdm
class = classes(:,1);
cDict = terms;

% mode = 'doc';
mode = 'basis';
scaleNum = 4;   % mode = 'basis'
funcNum = 2;    % mode = 'basis'
catNum = 2;     % mode = 'class'
docNum = 5;

if (strcmp(mode,'class')),
    catFreq = sum(X(class==catNum,:),1);
    
elseif (strcmp(mode,'basis')),
    basisFunc = G.Tree{scaleNum,1}.ExtBasis(:,funcNum);
    catFreq = sum(X.*repmat(sqrt(basisFunc.^2),[1 size(X,2)]),1);

elseif (strcmp(mode,'doc')),
    catFreq = X(docNum,:);
    fprintf(1,'Doc name: %d\n', classes(docNum,2));
end;

[ff,II] = sort(catFreq,'descend');
clipString = '';
for ii = 1:150, 
    % fprintf(1,'%s:%3.2f\n',cDict{II(ii)},ff(ii)*100); 
    tmpString = sprintf('%s:%3.2f\n',cDict{II(ii)},ff(ii).*100); 
    clipString = sprintf('%s%s',clipString,tmpString);
    % clipString = [clipString tmpString];
end;

% Copying results onto the clipboard for pasting into wordle advanced
clipboard('copy',clipString);


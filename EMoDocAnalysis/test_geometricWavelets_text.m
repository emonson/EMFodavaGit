%function [gW, Data] = test_geometricWavelets_MNIST_digits(GWTopts)

clear all;
close all;

%% the digits dataset

GWTopts = struct();
GWTopts.AmbientDimension = 1e12; % projection dimension

% geometric wavelets parameters
GWTopts.knn = 100;
GWTopts.knnAutotune = 40;
GWTopts.smallestMetisNet = 20;

% parameters for geometric wavelets
GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following two fields:

GWTopts.errorType = 'relative';  
GWTopts.threshold0 = [0.9 0.8];
GWTopts.precision  = 1e-2;

% The following thresholds are used in the code construct_geometricWavelets.m
GWTopts.threshold1 = 1e-1; % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = 5e-2; % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}

% The following parameter .pruning determines which version of geometric wavelets to use
GWTopts.pruning = 1;

% =1, plain construction without pruning
% =1.1, minimal height, node+grandchildren (deleting children) vs node+children+grandchildren, in terms of encoding cost
% =1.2, sparsity-preserving dictionary compression
% =1.3, minimal encoding-cost pruning
% =1.4, orthogonal geometric wavelets

% whether to use best approximations
GWTopts.addTangentialCorrections = true;

% whether to sparsify the scaling functions and wavelet bases
GWTopts.sparsifying = false;

% whether to split the wavelet bases into a common intersection and
% children-specific parts
GWTopts.splitting = false;


global gW X

% SciNews docs (original)
% Science News Articles dataset
% This version is for data that I re-tokenized with nltk

% data_dir = '/Users/emonson/Data/Fodava/EMoDocMatlabData';
% data_name = 'tdm_emo1nvj_112509.mat';
data_dir = '/Users/emonson/Data/Fodava/MauroDocAnalysis/Data';
data_name = 'X20.mat';
cd(data_dir);

load(data_name);

labels = dlmread('pure.classes');
fid = fopen('LabelsMeaning.txt');
articlegroups = textscan(fid,'%d = %s');
fclose(fid);

% Get rid of any "non-pure" classes (assigned 0 in X20)
%   by using filename integer converted to array index
%   Note: this takes care of missing 10976 file...
% tdm = tdm(:,classes(:,2)-10000+1);
X(classes(:,1)==0,:) = [];
X0 = X;

% ?? Cheat for GUI
imR = 28;
imC = 28;

% FREY FACES
% load('/Users/emonson/Data/Fodava/frey_rawface.mat');
% X0 = double(fliplr(ff'));
% imR = 20;
% imC = 28;

% % OLIVETTI FACES
% load('/Users/emonson/Data/Fodava/olivettifaces.mat');
% X0 = double(fliplr(faces'));
% imR = 64;
% imC = 64;
% % Rotate images for GUI
% yy = reshape(X0,[],imR,imC);
% yy1 = permute(yy,[1 3 2]);
% yy2 = flipdim(yy1,2);
% X0 = reshape(yy2,[],imR*imC);
% clear('yy','yy1','yy2');

WRITE_OUTPUT = false;
data_dir = '/Users/emonson/Data/Fodava/EMoGWDataSets';
out_file = 'text_20101118.mat';


N = size(X0,1);

cm = mean(X0,1);
X = X0 - repmat(cm, N,1);

if (GWTopts.AmbientDimension > size(X,2))
    GWTopts.AmbientDimension = size(X,2);
end

if GWTopts.AmbientDimension<size(X,2),
    [U,S,V] = svd(X,0);
    X = X*V(:,1:GWTopts.AmbientDimension);
else
    V = eye(size(X,2));
end;

gW.X = X;
gW.X_clean = X;
gW.X0 = X0;

% X= X0;
% [N,D] = size(X);

%% plot the data 

figure; do_plot_data(X(:,1:3));

%% construct geometric wavelets
gW = geometricWavelets(X,GWTopts);

%% Computing all wavelet coefficients 
Data = GWT_trainingData(gW, X);
Data.Xmean = cm;
Data.V=V;

%% Display the coefficents
GWT_DisplayCoeffs( gW, Data );

%% Plot approximation error
GWT_DisplayApproxErr( gW, Data );

%% Plot the reconstructed manifold at all scales
J = max(gW.Scales); % number of scales

% i = 39; % when digit =1
% 
% figure;
% for j = J:-1:1
%     X_approx = Data.Projections(:,:,j);
%     X_img = X_approx*V(:,1:GWTopts.AmbientDimension)'+repmat(cm, size(X_approx,1),1);
%     subplot(2,ceil((J+2)/2),j); imagesc(reshape(X_img(i,:), imC,[]))
%     set(gca, 'xTick', [5 15 25])
%     title(num2str(j)); colormap gray
% end
% 
% %% original but projected
% X_orig = X*V(:,1:GWTopts.AmbientDimension)'+repmat(cm, size(X,1),1);
% subplot(2,ceil((J+2)/2),J+1); imagesc(reshape(X_orig(i,:), imC,[]))
% title 'projection'
% set(gca, 'xTick', [5 15 25])
% colormap gray
% 
% %% original
% subplot(2,ceil((J+2)/2),J+2); imagesc(reshape(X0(i,:), imC,[]))
% title 'original'
% set(gca, 'xTick', [5 15 25])
% colormap gray
% 
% %%
% leafNode = gW.IniLabels(i);
% chain = dpath(gW.cp, leafNode);
% 
% figure;
% for i = 1:length(chain)
%     subplot(2,ceil(length(chain)/2),i);imagesc(reshape(Data.V(:,1:GWTopts.AmbientDimension)*gW.ScalFuns{chain(i)}, imC,[])); colormap gray
%     title(num2str(i));
%     colormap(map2);
%     balanceColor;
% end

%% Saving quantities for GUI use

if (WRITE_OUTPUT)
    cd(data_dir);
    
    % "poofing" out variables for easier loading in Python code 
    % and for file size since don't need nearly all of this stuff...
    % I know it makes it less flexible later...

    AmbientDimension = GWTopts.AmbientDimension;
    X = gW.X;
    % cm
    % imR
    % imC
    
    % Redundant for now...
    CelWavCoeffs = Data.CelWavCoeffs;
    
    num_nodes = length(gW.cp);
    LeafNodesImap(gW.LeafNodes) = 1:length(gW.LeafNodes);
    
    CelScalCoeffs = Data.CelScalCoeffs;
    
    % Should calculate Projections rather than storing -- it's big...
    
    % node_idx = leafNodes(leaf_node_idx);
    % data_idxs = find(gW.IniLabels == node_idx); % same as PointsInNet{net}
    % nPts = length(data_idxs);
    % j_max = gW.Scales(node_idx);
    % gWCentersnet = repmat(gW.Centers{node_idx},nPts,1);
    % Data.Projections(data_idxs,:,j_max) = Data.CelScalCoeffs{i,j_max}*gW.ScalFuns{node_idx}' + gWCentersnet;
    % X_approx = Data.Projections(:,:,scale);
    % X_img = X_approx*V(:,1:GWTopts.AmbientDimension)'+repmat(cm, size(X_approx,1),1);

    % Projections = Data.Projections;
    
    % May be able to get away with only saving
    % V(:,1:GWTopts.AmbientDimension)
    V = Data.V(:,1:GWTopts.AmbientDimension);
    
    cp = gW.cp;
    IniLabels = gW.IniLabels;
    PointsInNet = gW.PointsInNet;
    NumberInNet = gW.Sizes;
    ScalFuns = gW.ScalFuns;
    WavBases = gW.WavBases;
    Centers = gW.Centers;
    Scales = gW.Scales;
    IsALeaf = gW.isaleaf;
    LeafNodes = gW.LeafNodes;
    EigenVecs = gW.G.EigenVecs;
    EigenVals = gW.G.EigenVals;

    % ?? Cheat for GUI
    V = V(1:imR*imC,1:imR*imC);
    for ii = 1:length(ScalFuns),
        ScalFuns{ii} = ScalFuns{ii}(1:imR*imC,:);
        WavBases{ii} = WavBases{ii}(1:imR*imC,:);
        Centers{ii} = Centers{ii}(1:imR*imC);
    end
    X = X(:,1:imR*imC);
    cm = cm(1:imR*imC);
    cat_labels = labels(:,2);

    % Taking out saving GWTopts for now since gives problems on
    % scipy.io.loadmat when running from a standalone mac app...
    % Taking out Node... just for space since don't use them right now...
    save(out_file,...
        'cm','imR','imC','AmbientDimension','X','CelWavCoeffs',...
        'LeafNodesImap','CelScalCoeffs',...
        'V','cp','IniLabels','cat_labels',...
        'PointsInNet','NumberInNet','ScalFuns','WavBases',...
        'Centers','Scales','IsALeaf','LeafNodes','LeafNodesImap',...
        'EigenVecs','EigenVals');

end


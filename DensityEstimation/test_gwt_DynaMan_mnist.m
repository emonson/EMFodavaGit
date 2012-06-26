% Script for testing dynamic/manifold and a cusp singularity that grows at an unkown location of a manifold based on the GRMA framework

% close all;
% clear all;

%% Go parallel
if matlabpool('size')==0,
    matlabpool
end;

pQuantile = 0.1;                        % This is used as a local parameter to detect anomalies

%% Generate training data set
% DataSet.name = 'SwissRoll';
% DataSet.opts = struct('NumberOfPoints',2000,'EmbedDim',50);

fprintf('creating training data...\n');
DataSet.name = 'BMark_MNIST';

n_train = 2000;
n_test = 100;
n_total = n_train + n_test;

rand_idxs = randperm(n_total);
idxs_test = rand_idxs(1:n_test);
idxs_train = rand_idxs(n_test+1:end);

train_digits = [2 3 5 6 9];
anomalous_digits = [0];
INCLUDE_ANOMALOUS = true;
RANDOMIZE_ANOMALOUS = false;
REDUCED_DIMENSIONALITY = 100;
NOISE_LEVEL = 0.0;

X_train_sets = cell(1, length(train_digits));
X_test_sets = cell(1, length(train_digits) + length(anomalous_digits));

for ii = 1:length(train_digits),
    DataSet.opts = struct('NumberOfPoints', n_total, 'MnistOpts', struct('Sampling', 'RandN', 'QueryDigits', train_digits(ii), 'ReturnForm', 'vector'));
    tmp = GenerateDataSets( DataSet.name, DataSet.opts );
    if NOISE_LEVEL > 0,
        tmp = tmp + NOISE_LEVEL*randn(size(tmp));
    end;
    X_train_sets{ii} = tmp(:,idxs_train);
    X_test_sets{ii} = tmp(:,idxs_test);
end;

if INCLUDE_ANOMALOUS,
    for ii = 1:length(anomalous_digits),
        DataSet.opts = struct('NumberOfPoints', n_test, 'MnistOpts', struct('Sampling', 'RandN', 'QueryDigits', anomalous_digits(ii), 'ReturnForm', 'vector'));
        tmp = GenerateDataSets( DataSet.name, DataSet.opts );
        if RANDOMIZE_ANOMALOUS,
            n_pixels = size(tmp,1);
            for jj = 1:size(tmp,2),
                if true, % rand() < 1,
                    rand_idxs = randperm(n_pixels);
                    tmp_vec = tmp(rand_idxs,jj);
                    tmp(:,jj) = tmp_vec;
                end;
            end
        end;
        if NOISE_LEVEL > 0,
            tmp = tmp + NOISE_LEVEL*randn(size(tmp));
        end;
        X_test_sets{ii+length(train_digits)} = tmp;
    end;
end;

if REDUCED_DIMENSIONALITY > 0,
    % Use reduced-dimensionality data
    n_train_pts = size(cell2mat(X_train_sets),2);
    X0 = cat(2, cell2mat(X_train_sets), cell2mat(X_test_sets));

    cm = mean(X0,2);
    X = X0 - repmat(cm,1,size(X0,2));
    [~,S,V] = randPCA(X, REDUCED_DIMENSIONALITY);
    X2 = S*V';

    X_train = X2(:,1:n_train_pts);
    X_val = X2(:,(n_train_pts+1):end);
else
% Use original data
    X_train = cell2mat(X_train_sets);
    X_val = cell2mat(X_test_sets);
end;

clear('X_train_sets','X_test_sets','tmp', 'X', 'X0', 'X0_val', 'S', 'V');

%% Create the GWT and SVD models (this is fast enough)

% Set GMRA parameters (original script_DynamicManifold values)
GWTopts = struct('GWTversion',0);
GWTopts.ManifoldDimension = 2;
GWTopts.threshold1 = 1e-3;
GWTopts.threshold2 = .1;
GWTopts.addTangentialCorrections = true;
GWTopts.sparsifying = false;
GWTopts.splitting = false;
GWTopts.knn = 30;
GWTopts.knnAutotune = 20;
GWTopts.smallestMetisNet = 10;
GWTopts.verbose = 1;
GWTopts.shrinkage = 'hard';
GWTopts.avoidLeafnodePhi = false;
GWTopts.mergePsiCapIntoPhi  = true;
GWTopts.coeffs_threshold = 0;
GWTopts.errorType = 'relative';
GWTopts.threshold0 = 0.5;
GWTopts.precision  = 1e-4;

% These are the values used in the GWT RunExamples code...
% GWTopts.threshold1 = sqrt(2)*(1-cos(pi/24));    % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
% GWTopts.threshold2 = sqrt(2)*sin(pi/24);        % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
% GWTopts.addTangentialCorrections = false;
% GWTopts.sparsifying = false;
% GWTopts.sparsifying_method = 'ksvd'; % or 'spams'
% GWTopts.splitting = false;
% GWTopts.knn = 30;
% GWTopts.knnAutotune = 20;
% GWTopts.smallestMetisNet = 10;
% GWTopts.verbose = 1;
% GWTopts.shrinkage = 'hard';
% GWTopts.avoidLeafnodePhi = false;
% GWTopts.mergePsiCapIntoPhi  = false;
% GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:
% GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
% GWTopts.errorType = 'relative';
% GWTopts.precision  = .050; % only for leaf nodes

% Do GMRA
Timings.GMRA = cputime;
gMRA = GMRA(X_train, GWTopts);
DataGWT = FGWT(gMRA, X_train);
Timings.GMRA = cputime - Timings.GMRA;

%% Build a family of models, one per GWT scale, for the density
fprintf('\n Constructing GWT models...');

% Allocate memory
nScales = max(gMRA.Scales);
TimingsGWT_DensEst = zeros(nScales,1);
DensEst_GMRA = cell(1, nScales);

for j = 1:nScales,
    % Estimate density at scale j
    fprintf('\n Model at scale %d...',j);
    TimingsGWT_DensEst(j) = cputime;
    opts = struct('j', j, 'DensityEstimationMode', 'ScalingWaveletJoint');
    opts.Normalization = {'none', 'none'}; 
    opts.Normalization = {'zscore', 'zscore'};     % Both of these options seem to work...
    DensEst_GMRA{j} = GMRA_MeasureEstimate( gMRA, X_train, DataGWT, opts);
    TimingsGWT_DensEst(j) = cputime-TimingsGWT_DensEst(j);
end;

Timings.GWT_DensEst = TimingsGWT_DensEst; clear TimingsGWT_DensEst;

%% Now validate the model on a different random sample from the data set
% TBD: this should be compared to the training set above and used as calibration of the estimators

DataGWT_val = FGWT(gMRA, X_val);

fprintf('\n\n Running models on validation data...');

TimingsGWT_DensVal = zeros(nScales,1);
LogL_Val = zeros(1, nScales);
GMRA_Measure_Val = cell(1,nScales);

for j = 1:nScales,
    fprintf('\n Validation at scale %d...',j);
    TimingsGWT_DensVal(j) = cputime;
    
    [GMRA_Measure_Val{j}, LogL_Val(j)] = GMRA_MeasureEvaluate( DensEst_GMRA{j}, X_val, struct('mode', 'GMRA', 'GMRA', gMRA) );
    
    TimingsGWT_DensVal(j) = cputime-TimingsGWT_DensVal(j);
end;

Timings.GWT_DensVal = TimingsGWT_DensVal; 
clear('TimingsGWT_DensVal');

opts = struct('quantile', pQuantile);
opts.ratio = DensEst_GMRA;

LLmat = GMRA_DisplayMeasureEval( gMRA, DataGWT_val, GMRA_Measure_Val, opts );

fprintf('done.');

matF = LLmat;

figure;
imagesc(LLmat);
colormap(gray);
colorbar;
title('Log-Likelihood');
xlabel('points'); 
ylabel('scales');

%% Automatic selection of optimal scale for anomaly detection
matF_sort = sort(matF,2,'ascend');
infloc = isinf(matF_sort);
matF_sort(infloc)= 0;
matF_sort(infloc) = min(min(matF_sort));
meanlogL = mean(matF_sort(:,1:round(DataSet.opts.NumberOfPoints/100)),2);
[~,jopt] = min(meanlogL);

figure; 
plot(meanlogL, '-*'); 
hold on; 
plot(jopt,meanlogL(jopt), 'ro', 'MarkerSize', 12);
title('mean logL versus scale');

figure; 
scatter3(X_val(1,:), X_val(2,:), X_val(3,:), 10, LLmat(jopt,:), 'filled');
colorbar;
title('data set colored by logL');

fprintf('\n');

return;

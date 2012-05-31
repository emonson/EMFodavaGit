% Script for detecting a cusp singularity 
% that grows at an unkown location of a manifold
% based on the GRMA framework

close all;
clear all;

%% Go parallel
if matlabpool('size')==0,
    matlabpool
end;

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
anomalous_digits = [1];
INCLUDE_ANOMALOUS = true;
RANDOMIZE_ANOMALOUS = true;

X_train_sets = cell(1, length(train_digits));
X_test_sets = cell(1, length(train_digits) + length(anomalous_digits));

for ii = 1:length(train_digits),
    DataSet.opts = struct('NumberOfPoints', n_total, 'MnistOpts', struct('Sampling', 'RandN', 'QueryDigits', train_digits(ii), 'ReturnForm', 'vector'));
    tmp = GenerateDataSets( DataSet.name, DataSet.opts );
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
        X_test_sets{ii+length(train_digits)} = tmp;
    end;
end;
    
X_train = cell2mat(X_train_sets);
X_val = cell2mat(X_test_sets);
clear('X_train_sets','X_test_sets','tmp');

%% Create the GWT and SVD models (this is fast enough)

% Set GMRA parameters (original script_test_blip values)
% GWTopts = struct('GWTversion',0);
% GWTopts.ManifoldDimension = 2;
% GWTopts.threshold1 = 1e-3;
% GWTopts.threshold2 = .1;
% GWTopts.addTangentialCorrections = true;
% GWTopts.sparsifying = false;
% GWTopts.splitting = false;
% GWTopts.knn = 30;
% GWTopts.knnAutotune = 20;
% GWTopts.smallestMetisNet = 10;
% GWTopts.verbose = 1;
% GWTopts.shrinkage = 'hard';
% GWTopts.avoidLeafnodePhi = false;
% GWTopts.mergePsiCapIntoPhi  = true;
% GWTopts.coeffs_threshold = 0;
% GWTopts.errorType = 'relative';
% GWTopts.threshold0 = 0.5;
% GWTopts.precision  = 1e-4;

% These are the values used in the GWT RunExamples code...
GWTopts.threshold1 = sqrt(2)*(1-cos(pi/24));    % threshold of singular values for determining the rank of each ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.threshold2 = sqrt(2)*sin(pi/24);        % threshold for determining the rank of intersection of ( I - \Phi_{j,k} * \Phi_{j,k} ) * Phi_{j+1,k'}
GWTopts.addTangentialCorrections = false;
GWTopts.sparsifying = false;
GWTopts.sparsifying_method = 'ksvd'; % or 'spams'
GWTopts.splitting = false;
GWTopts.knn = 30;
GWTopts.knnAutotune = 20;
GWTopts.smallestMetisNet = 10;
GWTopts.verbose = 1;
GWTopts.shrinkage = 'hard';
GWTopts.avoidLeafnodePhi = false;
GWTopts.mergePsiCapIntoPhi  = false;
GWTopts.ManifoldDimension = 0; % if 0, then determine locally adaptive dimensions using the following fields:
GWTopts.threshold0 = 0.5; % threshold for choosing pca dimension at each nonleaf node
GWTopts.errorType = 'relative';
GWTopts.precision  = .050; % only for leaf nodes

%% Compute GWT and the transform of the data
fprintf('\n Computing GMRA and associated transforms...');

Timings.GMRA = cputime;
gMRA = GMRA(X_train, GWTopts);
DataGWT = FGWT(gMRA, X_train);
[Projections, tangentialCorrections] = IGWT(gMRA, DataGWT);
Timings.GMRA = cputime-Timings.GMRA;

%% Build a family of models, one per GWT scale, for the density

% Allocate memory
nScales = max(gMRA.Scales);
TimingsGWT_DensEst = zeros(nScales,1);
DensEst_GMRA = cell(1, nScales);

fprintf('\n Constructing GWT models...');
for j = 1:nScales,
    % Estimate density at scale j
    fprintf('\n Model at scale %d...',j);
    TimingsGWT_DensEst(j) = cputime;
    DensEst_GMRA{j} = GWT_EstimateDensityAtFixedScale( gMRA, X_train, DataGWT, j, ...
       struct('DensityEstimationMode', 'ScalingWaveletJoint')); %'ScalingWaveletJoint')); %'ScalingSpacesOnly'
    TimingsGWT_DensEst(j) = cputime-TimingsGWT_DensEst(j);
    
    % Now compute the cost of using j scales
    %subtree_idxs{j}     = [DensEst_GMRA{j}.cp_idx,get_ancestors(gMRA.cp,DensEst_GMRA{j}.cp_idx)];
    %GWTCost(j)          = computeDictionaryCost( gMRA, subtree_idxs{j}(:) )*(1+size(X,1)/size(X,2));
end;

Timings.GWT_DensEst = TimingsGWT_DensEst;

%% Now perform validation of the models

fprintf('\n Validating GMRA models...');

% perform GWT
DataGWT_val = FGWT(gMRA, X_val); 

matF = zeros(nScales, size(X_val,2));

% Compute log likelihood of the validation data
for j = 1:nScales,

    fprintf('\n Scale %d ...', j);

    [LogL(j), LogLPerPlan{j}, ~] = EvalLLFromMultiplePlaneDens...
        ( DensEst_GMRA{j}, X_val, struct('mode', 'GMRA', 'GMRA', gMRA, 'DataGWT', DataGWT_val) );

    for k = 1:length(LogLPerPlan{j}),
        if DensEst_GMRA{j}.child_idx(k) > 0, % nonleaf node
            matF(j, DataGWT_val.PointsInNet{DensEst_GMRA{j}.child_idx(k)}) = LogLPerPlan{j}{k};
        else
            matF(j, DataGWT_val.PointsInNet{DensEst_GMRA{j}.cp_idx(k)})    = LogLPerPlan{j}{k};
        end;
    end;

end;

LLmat = matF;

figure;
imagesc(LLmat);
colormap(gray);
colorbar;
title('Log-Likelihood');
xlabel('points'); 
ylabel('scales');

% sort matF according to leaf nodes
[val_sort, idx_sort] = sort(DataGWT_val.leafNodeLabels);
matF = matF(:, idx_sort);

% automatic selection of optimal scale for anomaly detection
matF_sort = sort(matF,2,'ascend');
infloc = isinf(matF_sort);
matF_sort(infloc)= 0;
matF_sort(infloc) = min(min(matF_sort));
meanlogL = mean(matF_sort(:,1:round(DataSet.opts.NumberOfPoints/100)),2);
[~,jopt] = min(meanlogL); 

figure; 
plot(meanlogL, '-*'); 
hold on; 
plot(jopt, meanlogL(jopt), 'ro', 'MarkerSize', 12);
title('mean logL versus scale');

fprintf('\n');


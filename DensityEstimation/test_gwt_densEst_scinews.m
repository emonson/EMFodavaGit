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
DataSet.name = 'ScienceNews';

[X, GWTopts, imgOpts] = GenerateData_and_SetParameters(DataSet.name);

labels = imgOpts.Labels;
[YY,II] = sort(labels);
labels = labels(II);
X = X(:,II);

% randomly take percentage of document for training
rand_idxs = rand([1 size(X,2)]) > 0.05;

% these are the categories for training
cat_idxs = labels ~= 1;

X_train = X(:, (rand_idxs & cat_idxs));
labels_train = labels((rand_idxs & cat_idxs));

% take the rest of the document for validation
X_val = X(:, ~(rand_idxs & cat_idxs));
labels_val = labels(~(rand_idxs & cat_idxs));

DataSet.opts = struct('NumberOfPoints', length(labels_val));

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


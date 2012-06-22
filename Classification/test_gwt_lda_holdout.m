% clear all
% close all
% clc

%% Go parallel
if matlabpool('size')==0,
    matlabpool
end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits','YaleB_Faces','croppedYaleB_Faces','ScienceNews',...
                  'Medical12images','Medical12Sift','CorelImages','CorelSift',...
                  'Olivetti_faces',...
                  '20NewsSubset1','20NewsSubset2tf','20NewsSubset3','20NewsSubset4', ...
                  '20NewsCompSetOf5'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

pGWTversion = 0;

% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = EMo_GenerateData_and_SetParameters(pExampleNames{pExampleIdx});

% Construct geometric wavelets
GWTopts.GWTversion = pGWTversion;
fprintf(1, '\nGMRA\n\n');
GWT = GMRA(X, GWTopts);

% Compute all wavelet coefficients 
fprintf(1, 'GWT Training Data\n\n');
[GWT, Data] = GWT_trainingData(GWT, X);

% Deleting some unneeded data for memory's sake
% Data = rmfield(Data,'Projections');
% Data = rmfield(Data,'TangentialCorrections');
Data = rmfield(Data,'MatWavCoeffs');

%% Test original data

data_set = pExampleNames{pExampleIdx};

% [data, labels] = lda_generateData(data_set, 'dim', 30, 'digits', [1 2 3], 'n_ea_digit', 1000);

labels = imgOpts.Labels;

% [total_errors, std_errors] = lda_crossvalidation( GWT.X, labels );

n_pts = length(labels);
n_cats = length(unique(labels));

% fprintf(1, '\nOriginal\n');
% fprintf(1, '\nData set: %s\n', data_set);
% fprintf(1, 'Categories: %d, Data points: %d\n', n_cats, n_pts);
% fprintf(1, 'Avg Accuracy: %3.2f\n', 1.0 - total_errors/n_pts);
% fprintf(1, 'Error Rate: %d / %d\n', total_errors, n_pts);
% fprintf(1, 'Standar dev: %3.2f\n\n', std_errors);

%% Test holdout data split for classifier accuracy measurement

% Combined uses both scaling functions and wavelets together for all fine
% scales. Otherwise, only scaling functions are used for all scales.
COMBINED = false;

results = struct();

for idx = 1:length(GWT.cp),
    
    if mod(idx, 100) == 0,
        fprintf(1, 'LDA cross-validation node %d of %d\n', idx, length(GWT.cp));
    end
    
    if ~COMBINED || idx == length(GWT.cp),
        coeffs = cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == idx})';
    else
        coeffs = cat(2, cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == idx}), cat(1,Data.CelWavCoeffs{Data.Cel_cpidx == idx}))';
    end
    dataIdxs = GWT.PointsInNet{idx};
    dataLabels = imgOpts.Labels(dataIdxs);

    n_pts = length(dataLabels);
    n_cats = length(unique(dataLabels));

    if (n_cats > 1 && n_pts > 1),
        [total_errors, std_errors] = lda_crossvalidation( coeffs, dataLabels );
        results(idx).total_errors = total_errors;
        results(idx).std_errors = std_errors;
    else
        results(idx).total_errors = inf;
        results(idx).std_errors = inf;
    end
    
%     fprintf(1, 'Scale %d\n', GWT.Scales(idx));
%     fprintf(1, 'Categories: %d, Data points: %d\n', n_cats, n_pts);
%     fprintf(1, 'Avg Accuracy: %3.2f\n', 1.0 - total_errors/n_pts);
%     fprintf(1, 'Error Rate: %d / %d\n', total_errors, n_pts);
%     fprintf(1, 'Standar dev: %3.2f\n\n', std_errors);

end




%% Tree of results
% http://stackoverflow.com/questions/5065051/add-node-numbers-get-node-locations-from-matlabs-treeplot

H = figure;
treeplot(GWT.cp, 'k.', 'c');

% treeplot is limited with control of colors, etc.
P = findobj(H, 'Color', 'c');
set(P, 'Color', [247 201 126]/255);

count = size(GWT.cp,2);
[x,y] = treelayout(GWT.cp);
x = x';
y = y';
error_array = [results(:).total_errors];
error_strings = cellstr(num2str(error_array'));
std_array = [results(:).std_errors];
std_strings = cellstr(num2str(round(std_array)'));
nptsinnode_strings = cellstr(num2str((cellfun(@(x) size(x,2),GWT.PointsInNet))'));

childerr = zeros(length(error_array), 1);
childstd = zeros(length(std_array), 1);
for ii = 1:length(childerr),
   childerr(ii) = sum(error_array(GWT.cp == ii));
   childstd(ii) = sum(std_array(GWT.cp == ii));
end
childerr_strings = cellstr(num2str(childerr));
childstd_strings = cellstr(num2str(round(childstd)));

combo_strings = strcat(error_strings, '~', std_strings);
childcombo_strings = strcat(childerr_strings, '~', childstd_strings);

text(x(:,1), y(:,1), combo_strings, ...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
text(x(:,1), y(:,1), childcombo_strings, ...
    'VerticalAlignment','top','HorizontalAlignment','left','Color',[0.6 0.2 0.2])
title({['LDA cross validation: ' strrep(data_set, '_', ' ') ' - ' num2str(n_pts) ' pts']}, ...
    'Position', [0.01 1.02], 'HorizontalAlignment', 'Left', 'Margin', 10);

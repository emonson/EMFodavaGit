% clear all
% close all
% clc

%% Go parallel
% if matlabpool('size')==0,
%     matlabpool
% end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits','YaleB_Faces','croppedYaleB_Faces','ScienceNews'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

pGWTversion = 0;

% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = GenerateData_and_SetParameters(pExampleNames{pExampleIdx});

% Construct geometric wavelets
GWTopts.GWTversion = pGWTversion;
GWT = GMRA(X, GWTopts);

% Compute all wavelet coefficients 
[GWT, Data] = GWT_trainingData(GWT, X);


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

figure;
treeplot(GWT.cp, 'k.', 'c');
count = size(GWT.cp,2);
[x,y] = treelayout(GWT.cp);
x = x';
y = y';
errors_array = [results(:).total_errors];
numerrors = cellstr(num2str(errors_array'));
nptsinnode = cellstr(num2str((cellfun(@(x) size(x,2),GWT.PointsInNet))'));
childerrors = zeros(1,length(numerrors));
for ii = 1:length(childerrors),
   childerrors(ii) = sum(errors_array(GWT.cp == ii)); 
end
childerrvalues = cellstr(num2str(childerrors'));
text(x(:,1), y(:,1), numerrors, 'VerticalAlignment','bottom','HorizontalAlignment','right')
text(x(:,1), y(:,1), childerrvalues, 'VerticalAlignment','top','HorizontalAlignment','left','Color',[0.6 0.2 0.2])
% title({['LDA cross validation: ' data_set]});

stream0 = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(stream0);

%% Go parallel
% if matlabpool('size')==0,
%     matlabpool('OPEN',6);
% end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits','YaleB_Faces','croppedYaleB_Faces','ScienceNews', ...
                  'Medical12images','Medical12Sift','CorelImages','CorelSift', ...
                  'Olivetti_faces', ...
                  '20NewsAllTrain', '20NewsAllTest', '20NewsAllCombo', ...
                  '20NewsSubset1','20NewsSubset2tf','20NewsSubset3','20NewsSubset4', ...
                  '20NewsCompSetOf5'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = EMo_GenerateData_and_SetParameters(pExampleNames{pExampleIdx});

% Special case for YaleB data, grabbing Subject label
if (size(imgOpts.Labels,1) > 1),
    imgOpts.Labels = imgOpts.Labels(3,:);
end
fprintf(1, '\n\n');

% Dimension limits to loop through
dim_limit = [5000 2000 1000 800 500 200 100 50 20 10 5 2];

% If orig dim is reasonable, put in randPCA version of orig
if size(X,1) < 1000,
    dim_limit = cat(2, dim_limit, size(X,1));
    dim_limit = sort(dim_limit, 'descend');
end
n_dim_trials = length(dim_limit);

errors = zeros(n_dim_trials, 1);
dims = zeros(n_dim_trials, 1);
stds = zeros(n_dim_trials, 1);

% Outer loop over holdout groups
for ii = 1:n_dim_trials,
    
    % Dimensionality into which to project data for straight LDA (0 = no dim reduction)
    if size(X,1) >= dim_limit(ii),
        straight_lda_dim = dim_limit(ii);
        % Dim must be smaller than number of points
        if size(X,2) < straight_lda_dim,
            straight_lda_dim = size(X,2) - 1;
        end
    else
        straight_lda_dim = 0;
    end
    
    X0 = X;
    cm = mean(X0,2);
    X1 = X0 - repmat(cm, 1, size(X0,2));
    if ((straight_lda_dim > 0) && (straight_lda_dim <= size(X,1))),
        % clear('X0');
        fprintf('pre-LDA randPCA from %d to %d dimensions\n', size(X1,1), straight_lda_dim);
        % NOTE: randPCA calls RAND
        [~,S,V] = randPCA(X1, straight_lda_dim);
        X_lda = S*V';
        % clear('X1', 'S', 'V');
    else
        X_lda = X1;
    end;
    
    fprintf(1, 'Straight LDA in %d dim\n', straight_lda_dim);
    [errors(ii), stds(ii)] = lda_multi_crossvalidation(X_lda, imgOpts.Labels);
    dims(ii) = size(X_lda, 1);
    % clear('X_lda');

end

figure;
semilogx(dims, errors, 'r.-');
ylim([0 1.1*max(errors)]);
title(['PCA-LDA-CV: ' strrep(pExampleNames{pExampleIdx}, '_', ' ') ' - ' num2str(size(X,2)) ' pts'], ...
    'Position', [1 1.11*max(errors)], 'HorizontalAlignment', 'Left', 'Margin', 10);
xlabel('dimensionality');
ylabel('average errors (10 trials of 5 holdout group crossvalidation)');


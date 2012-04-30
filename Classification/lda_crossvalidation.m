function [sum_errors, std_errors, errors_array] = lda_crossvalidation( data, labels, varargin )

% Cross-validation (1/5 holdout for now) of Linear Discriminant Analysis
%
% data = [d n] array of measurements
% labels = [1 n] array of category labels (integers)
% 
% (optional)
% holdout_groups = 5 : number of groups for cross-validation
%
% Requires Will Dwinnell's LDA code
%   http://matlabdatamining.blogspot.com/2010/12/linear-discriminant-analysis-lda.html
%   http://www.mathworks.com/matlabcentral/fileexchange/29673-lda-linear-discriminant-analysis

%% Utility function

    function [tf] = allintarray(xx)
        tf = false;
        % numeric
        if ~isnumeric(xx),
            return;
        end
        % 1d array
        if size(xx,1) ~= 1,
            return;
        end
        % integers
        for idx = 1:length(xx),
            if uint8(xx(idx)) ~= xx(idx),
                return;
            end
        end
        tf = true;
    end

%% Argument parsing

% Parameter value defaults
% dim = 0 is a flag for not doing any svd dimensionality reduction
holdout_groups = 5;

% Need value to be an integer, but isinteger() tests for int array...
checkHoldoutValue = @(x) (isnumeric(x) && (uint8(x) == x));
checkDigitsArray = @(x) allintarray(x);
    
p = inputParser;

addRequired(p, 'data', @isnumeric);
addRequired(p, 'labels', checkDigitsArray);
addParamValue(p, 'holdout_groups', holdout_groups, checkHoldoutValue);

parse(p, data, labels, varargin{:});

meas = p.Results.data;
cats = p.Results.labels;
m = p.Results.holdout_groups;

%% hold out 1/m

un_cats = unique(cats); 
n_labels = length(cats);

groups = randi(m, [n_labels 1]);

errors_array = zeros(m, 1);
for rr = 1:m,
    
    % train and test 
    meas_train = meas(:,groups ~= rr);
    meas_test = meas(:,groups == rr);
    labels_train = cats(groups ~= rr);
    labels_test = cats(groups == rr);
    n_labels_test = length(labels_test);
    
    % LDA code wants measurements [n d] order
    W = LDA(meas_train', labels_train');
    
    % Use the model on test set
    L = [ones(n_labels_test,1) meas_test'] * W';
    % P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);

    [~,I] = max(L,[],2);
    errors_array(rr) = sum(un_cats(I) ~= labels_test);
end

sum_errors = sum(errors_array);
std_errors = std(errors_array);


end

function [n_errors, model] = lda_traintest( data_train, labels_train, data_test, labels_test, varargin )

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

% Need value to be an integer, but isinteger() tests for int array...
checkDigitsArray = @(x) allintarray(x);
    
p = inputParser;

% NOTE: These have to be added in the arglist order!!
addRequired(p, 'data_train', @isnumeric);
addRequired(p, 'labels_train', checkDigitsArray);
addRequired(p, 'data_test', @isnumeric);
addRequired(p, 'labels_test', checkDigitsArray);

parse(p, data_train, labels_train, data_test, labels_test, varargin{:});

meas_train = p.Results.data_train;
meas_test = p.Results.data_test;
cats_train = p.Results.labels_train;
cats_test = p.Results.labels_test;

%% do lda on train and then test on same model

% un_cats = unique(cat(2, cats_train, cats_test));
un_cats_train = unique(cats_train);

% LDA code wants measurements [n d] order
model = LDA(meas_train', cats_train');

% Any points in the test set that have categories outside of the
% training categories has to be removed and just counted as wrong
% since the model wasn't built to handle those categories
extra_cats = setdiff(cats_test, cats_train);
sub_cats_test = cats_test;
sub_meas_test = meas_test;
pre_errors = 0;

for label = extra_cats,
    leave_out_bool = (sub_cats_test ~= label);
    sub_cats_test = sub_cats_test(leave_out_bool);
    sub_meas_test = sub_meas_test(:,leave_out_bool);
    pre_errors = pre_errors + sum(~leave_out_bool);
end
n_labels_test = length(sub_cats_test);

% Use the model on test set
L = [ones(n_labels_test,1) sub_meas_test'] * model';
% P = exp(L) ./ repmat(sum(exp(L),2),[1 size(L,2)]);

[~,I] = max(L,[],2);

n_errors = sum(un_cats_train(I) ~= sub_cats_test) + pre_errors;

end

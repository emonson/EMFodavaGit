% Test holdout data split for classifier accuracy measurement

data_set = 'sciencenews';

% [data, labels] = lda_generateData(data_set, 'dim', 30);
[data, labels] = lda_generateData( data_set, 'dim', 50, 'digits', [0 1 2 3 4 5 6 7 8 9], 'n_ea_digit', 10000);

[total_errors, std_errors] = lda_crossvalidation( data, labels, 'holdout_groups', 5 );

%% Results summary

n_pts = length(labels);
n_cats = length(unique(labels));

fprintf(1, '\nData set: %s\n', data_set);
fprintf(1, 'Categories: %d, Data points: %d\n', n_cats, n_pts);
fprintf(1, 'Avg Accuracy: %3.2f\n', 1.0 - total_errors/n_pts);
fprintf(1, 'Error Rate: %d / %d\n', total_errors, n_pts);
fprintf(1, 'Standar dev: %3.2f\n\n', std_errors);

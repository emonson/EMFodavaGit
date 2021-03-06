% clear all
% close all
% clc

stream0 = RandStream('mt19937ar','Seed',2);
RandStream.setGlobalStream(stream0);

%% Go parallel
% if matlabpool('size')==0,
%     matlabpool('OPEN',6);
% end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits_Full', 'MNIST_Digits_Subset','YaleB_Faces','croppedYaleB_Faces','ScienceNews', ...
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

pGWTversion = 0;

% Generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = EMo_GenerateData_and_SetParameters(pExampleNames{pExampleIdx});

% Construct geometric wavelets
GWTopts.GWTversion = pGWTversion;
fprintf(1, '\nGMRA\n\n');
GWT = GMRA(X, GWTopts);

% Get rid of this original since have copy in GWT.X
clear('X');

% Compute all wavelet coefficients 
fprintf(1, 'GWT Training Data\n\n');
[GWT, Data] = GWT_trainingData(GWT, GWT.X);

% Deleting some unneeded data for memory's sake
% Data = rmfield(Data,'Projections');
% Data = rmfield(Data,'TangentialCorrections');
if isfield(Data, 'MatWavCoeffs'),
    Data = rmfield(Data,'MatWavCoeffs');
end

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

% NOTE: For now testing out initializing all results so array will be
%  the right length later on, but init values = NaN and values tested but
%  too big = Inf
for ii = 1:length(GWT.cp),
    results(ii).self_error = NaN;
    results(ii).self_std = NaN;
    results(ii).best_children_errors = NaN;
    results(ii).direct_children_errors = NaN;
    results(ii).error_value_to_use = NaN;
end

% Version that tests holdout for whole tree in cp order
% for idx = 1:length(GWT.cp),
%     
%     if mod(idx, 100) == 0,
%         fprintf(1, 'LDA cross-validation node %d of %d\n', idx, length(GWT.cp));
%     end
%     
%     if ~COMBINED || idx == length(GWT.cp),
%         coeffs = cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == idx})';
%     else
%         coeffs = cat(2, cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == idx}), cat(1,Data.CelWavCoeffs{Data.Cel_cpidx == idx}))';
%     end
%     dataIdxs = GWT.PointsInNet{idx};
%     dataLabels = imgOpts.Labels(dataIdxs);
% 
%     n_pts = length(dataLabels);
%     n_cats = length(unique(dataLabels));
% 
%     if (n_cats > 1 && n_pts > 1),
%         [total_errors, std_errors] = lda_crossvalidation( coeffs, dataLabels );
%         results(idx).total_errors = total_errors;
%         results(idx).std_errors = std_errors;
%     else
%         results(idx).total_errors = inf;
%         results(idx).std_errors = inf;
%     end
%     
% %     fprintf(1, 'Scale %d\n', GWT.Scales(idx));
% %     fprintf(1, 'Categories: %d, Data points: %d\n', n_cats, n_pts);
% %     fprintf(1, 'Avg Accuracy: %3.2f\n', 1.0 - total_errors/n_pts);
% %     fprintf(1, 'Error Rate: %d / %d\n', total_errors, n_pts);
% %     fprintf(1, 'Standar dev: %3.2f\n\n', std_errors);
% 
% end

% Version that tests holdout by walking tree only down as far as it needs
% before it hits single class or too small nodes

tree_parent_idxs = GWT.cp;

% Container for child nodes which need to be freed up if a node eventually
% switches from USE_SELF to USE_CHILDREN, but some children have stopped
% propagating down the tree because they were past the ALLOWED_DEPTH of
% the indexed node
children_to_free = cell([length(GWT.cp) 1]);

% Since now including self in find should give empty if all self
% and parents are USE_CHILDREN, and 1 if self is USE_SELF, so
% allowed_depth should start at 1 
ALLOWED_DEPTH = 2;
                
% Flag for error status on each node
USE_SELF = 1;
UNDECIDED = -100;
USE_CHILDREN = -1;

% Start at root of the tree (cp(root_idx) == 0)
root_idx = find(tree_parent_idxs == 0);

if (length(root_idx) > 1)
    fprintf( 'cp contains too many root nodes (cp == 0)!! \n' );
    return;
else
    % This routine calculates errors for the children of the current node
    % so we need to first calculate the root node error
    [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data, imgOpts.Labels, root_idx, COMBINED );

    % Record the results for the root node
    results(root_idx).self_error = total_errors;
    results(root_idx).self_std = std_errors;
    results(root_idx).error_value_to_use = UNDECIDED;
    fprintf( 'current node: %d\n', root_idx );
    
    % The java deque works with First = most recent, Last = oldest
    %   so since it can be accessed with removeFirst / removeLast
    %   it can be used either as a LIFO stack or FIFO queue
    % Here I'm trying it as a deque/queue to do a breadth-first tree
    %   traversal
    node_idxs = java.util.ArrayDeque();
    node_idxs.addFirst(root_idx);

    % Main loop to work iteratively down the tree breadth first
    while (~node_idxs.isEmpty())
        current_node_idx = node_idxs.removeLast();
        fprintf( 'current node: %d\n', current_node_idx );
        
        % Get list of parent node indexes for use in a couple spots later
        % TODO: Move to a routine...
        current_parents_idxs = [];
        tmp_current_idx = current_node_idx;
        while (tree_parent_idxs(tmp_current_idx) > 0), 
            tmp_parent_idx = tree_parent_idxs(tmp_current_idx); 
            current_parents_idxs(end+1) = tmp_parent_idx; 
            tmp_current_idx = tmp_parent_idx; 
        end
        
        % Get children of the current node
        current_children_idxs = find(tree_parent_idxs == current_node_idx);
        
        % Loop through the children
        for current_child_idx = current_children_idxs,

            % Calculate the error on the current child
            [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data, imgOpts.Labels, current_child_idx, COMBINED );

            % Record the results for the current child
            results(current_child_idx).self_error = total_errors;
            results(current_child_idx).self_std = std_errors;
            results(current_child_idx).error_value_to_use = UNDECIDED;
            % fprintf( '\tchild node: %d\n', current_child_idx );
        end
        
        % If no children, want error to be infinite for any comparisons
        children_error_sum = Inf;
        % Set children errors to child sum (if there are children because sum([]) == 0)
        if ~isempty(current_children_idxs)
            children_error_sum = sum( [results(current_children_idxs).self_error] );
            results(current_node_idx).direct_children_errors = children_error_sum;
            results(current_node_idx).best_children_errors = children_error_sum;
        end
        
        % Compare children results to self error
        self_error = results(current_node_idx).self_error;
        % NOTE: Here is where to put some slop based on standard deviation
        if (self_error < children_error_sum)
            % Set status = USE_SELF
            results(current_node_idx).error_value_to_use = USE_SELF;
            
        else
            % Set status = USE_CHILDREN
            results(current_node_idx).error_value_to_use = USE_CHILDREN;
            
            % Propagate difference up parent chain
            error_difference = self_error - children_error_sum;
            fprintf('Node %d has %d error difference\n', current_node_idx, error_difference);
            % Loop through list of parent nodes
            for parent_node_idx = current_parents_idxs,
                
                % Subtract difference from best_children_errors
                fprintf('\tParent node %d best children error %d\n', parent_node_idx, results(parent_node_idx).best_children_errors);
                results(parent_node_idx).best_children_errors = results(parent_node_idx).best_children_errors - error_difference;
                fprintf('\t\tnow down to %d\n', results(parent_node_idx).best_children_errors);
                
                % If parent.status = USE_CHILDREN
                if (results(parent_node_idx).error_value_to_use == USE_CHILDREN)
                    % Propagate differnce up to parent
                    continue;
                
                % else if parent.status = USE_SELF
                elseif (results(parent_node_idx).error_value_to_use == USE_SELF)
                    % Compare best_children_errors to self_error
                    % NOTE: Here again use same slop test as above...
                    
                    % if still parent.self_error < parent.best_children_errors
                    if (results(parent_node_idx).self_error < results(parent_node_idx).best_children_errors),
                        % stop difference propagation
                        break;
                    % else if now parent.best_children_errors < parent.self_error
                    else
                        % parent.status = USE_CHILDREN
                        results(parent_node_idx).error_value_to_use = USE_CHILDREN;
                        % propagate this NEW difference up to parent
                        error_difference = results(parent_node_idx).self_error - results(parent_node_idx).best_children_errors;
                        % Since some children of this node might have
                        % not added their children to the queue because
                        % this node was too far up the tree for
                        % ALLOWED_DEPTH, now that this has switched, need
                        % to check those older nodes to see if now their
                        % children should be added...
                        for idx = children_to_free{parent_node_idx}
                            node_idxs.addFirst(idx);
                            fprintf(' * *   freeing: %d\n', idx);
                        end
                        children_to_free{parent_node_idx} = [];
                       continue;
                    end
                else
                    fprintf('\nERROR: parent error status flag not set properly on index %d!!\n', parent_node_idx);
                end
            end
        end

        % Allowing here to go a certain controlled depth beyond where
        %   the children seem to be worse than a parent to see if it
        %   eventually reverses. Set threshold to Inf to use whole tree
        %   of valid error values. Set threshold to zero to never go beyond
        %   a single reversal where children are greater than the parent.

        % Figure out how far up tree to highest USE_SELF
        % If hole_depth < hole_depth_threshold
            % Push children on to queue for further processing
        % else
            % stop going any deeper
        self_parent_idx_chain = [current_node_idx current_parents_idxs];
        self_parent_status_flags = [results(self_parent_idx_chain).error_value_to_use];
        use_self_depth = find(self_parent_status_flags == USE_SELF, 1, 'last');
        % Depth set with this test
        % Root node or not found gives empty find result
        
        use_self_depth_low_enough = isempty(use_self_depth) || (use_self_depth <= ALLOWED_DEPTH);
        
        % All children must have finite error sums to go lower in any child
        all_children_errors_finite = isfinite(children_error_sum);
        
        % If child errors are finite, but the node is too deep, keep track
        % of the child indexes to add to the queue in case the USE_SELF
        % node it's under switches to USE_CHILDREN
        if (~use_self_depth_low_enough && all_children_errors_finite)
            problem_parent_idx = self_parent_idx_chain(use_self_depth);
            children_to_free{problem_parent_idx} = cat(2, children_to_free{problem_parent_idx}, current_children_idxs);
        end
        
        % Only addFirst children on to the stack if this node qualifies
        if (use_self_depth_low_enough && all_children_errors_finite)
            % Find childrent of current node
            % DEBUG
            fprintf(' + + Current index: %d\n', current_node_idx);
            for idx = current_children_idxs
                node_idxs.addFirst(idx);
                fprintf(' + +   adding: %d\n', idx);
            end
        else
            % DEBUG
            fprintf(' - - Current index: %d\n', current_node_idx);
            fprintf(' - -   use_self_depth: %d, all_children_errors_finite: %d\n', use_self_depth_low_enough, all_children_errors_finite);
        end
    end
end


% Tree of results
% http://stackoverflow.com/questions/5065051/add-node-numbers-get-node-locations-from-matlabs-treeplot

H = figure;
treeplot(GWT.cp, 'k.', 'c');

% treeplot is limited with control of colors, etc.
P = findobj(H, 'Color', 'c');
set(P, 'Color', [247 201 126]/255);

% count = size(GWT.cp,2);
[x,y] = treelayout(GWT.cp);
x = x';
y = y';
hold();

% Show which nodes were used (self or children)
ee = [results(:).error_value_to_use];
use_self_bool = ee >= USE_SELF;
plot(x(use_self_bool), y(use_self_bool), 'r.', 'MarkerSize', 20);

error_array = round([results(:).self_error]);
error_strings = cellstr(num2str(error_array'));
std_array = round([results(:).self_std]);
std_strings = cellstr(num2str(std_array'));
% nptsinnode_strings = cellstr(num2str((cellfun(@(x) size(x,2), GWT.PointsInNet))'));
cp_idx_strings = cellstr(num2str((1:length(GWT.cp))'));

childerr_strings = cell(length(GWT.cp),1);
for ii = 1:length(GWT.cp),
    childerr_strings{ii} = num2str(round(results(ii).direct_children_errors));
end

% combo_strings = strcat(error_strings, '~', std_strings);
% childcombo_strings = strcat(childerr_strings, '~', childstd_strings);
combo_strings = error_strings;
childcombo_strings = childerr_strings;

besterr_strings = cell(length(GWT.cp),1);
for ii = 1:length(GWT.cp),
    besterr_strings{ii} = num2str(round(results(ii).best_children_errors));
end

% Node errors
% Only displaying the finite values for now
finite_values = isfinite(str2double(childerr_strings));
text(x(finite_values,1), y(finite_values,1), combo_strings(finite_values), ...
    'VerticalAlignment','bottom','HorizontalAlignment','right')
% Child node errors
text(x(finite_values,1), y(finite_values,1), childcombo_strings(finite_values), ...
    'VerticalAlignment','top','HorizontalAlignment','right','Color',[0.6 0.2 0.2])
% Best Child node errors
text(x(finite_values,1), y(finite_values,1), besterr_strings(finite_values), ...
    'VerticalAlignment','top','HorizontalAlignment','left','Color',[0.2 0.6 0.2])
% Node cp index
% text(x(:,1), y(:,1), cp_idx_strings, ...
%     'VerticalAlignment','bottom','HorizontalAlignment','left','Color',[0.2 0.2 0.6])

title({['LDACV: ' num2str(ALLOWED_DEPTH) ' deep ' strrep(data_set, '_', ' ') ' - ' num2str(n_pts) ' pts']}, ...
    'Position', [0.01 1.02], 'HorizontalAlignment', 'Left', 'Margin', 10);

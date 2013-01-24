function [total_errors, std_errors] = gwt_single_node_lda_crossvalidation( GWT, Data, imgOpts, current_node_idx, COMBINED )
% Uses GWT for cp and PointsInNet
% Uses Data for Cel_cpidx, CelScalCoeffs, CelWavCoeffs
% COMBINED is boolean for whether to use Scal/Wav combined or not as basis
%   for LDA

    if ~COMBINED || current_node_idx == length(GWT.cp),
        coeffs = cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx})';
    else
        coeffs = cat(2, cat(1, Data.CelScalCoeffs{Data.Cel_cpidx == current_node_idx}), cat(1,Data.CelWavCoeffs{Data.Cel_cpidx == current_node_idx}))';
    end
    dataIdxs = GWT.PointsInNet{current_node_idx};
    dataLabels = imgOpts.Labels(dataIdxs);

    node_pts = length(dataLabels);
    node_cats = length(unique(dataLabels));

    if (node_cats > 1 && node_pts > 1),
        % [total_errors, std_errors] = lda_multi_crossvalidation( coeffs, dataLabels );
        [total_errors, std_errors] = lda_multi_crossvalidation( coeffs, dataLabels );
    else
        total_errors = inf;
        std_errors = inf;
    end
    
end
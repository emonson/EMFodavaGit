function [Idxs,Cats] = docChooseScalePoints(classes,ExtIdxs,mode,N)
% [Idxs,Cats] = docChooseScalePoints(classes,ExtIdxs,mode,N)
% For document analysis
% Choose multiscale ExtBasis (ExtIdxs) points until have at least N of each class
% classes needs to be 1d vector of integer category labels 
%   (label ints don't need to be contiguous)
% 
% mode = 'duplicates', 'all'
%
% EMonson -- 28 Apr 2009

numPts = length(classes);

uniqueClasses = unique(classes);
counts = zeros(length(uniqueClasses),1);

if strcmpi(mode,'duplicates'),
    for ii = 1:numPts,
        idx = ExtIdxs(ii);
        Idxs(ii) = idx;
        Cats(ii) = classes(idx);
        logicalClass = (uniqueClasses == classes(idx));
        counts(logicalClass) = counts(logicalClass) + 1;
        if (min(counts) >= N), break; end;
    end;
elseif strcmpi(mode,'number'),
    if (N > length(ExtIdxs)),
        error(['docChooseScalePoints: ' int2str(N) 'chosen, but only ' int2str(length(ExtIdxs)) ' points in ExtIdxs']);
        return;
    end;
    for ii = 1:N,
        idx = ExtIdxs(ii);
        Idxs(ii) = idx;
        Cats(ii) = classes(idx);
    end;
elseif strcmpi(mode,'all'),
    for ii = 1:length(ExtIdxs),
        idx = ExtIdxs(ii);
        Idxs(ii) = idx;
        Cats(ii) = classes(idx);
        logicalClass = (uniqueClasses == classes(idx));
        counts(logicalClass) = counts(logicalClass) + 1;
    end;
else
    error(['docChooseScalePoints: ' mode ' not a valid mode']);
end;
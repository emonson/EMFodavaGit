function [Idxs,Cats] = docChooseRandomPoints(classes,mode,N)
% [Idxs,Cats] = docChooseRandomPoints(classes,N)
% For document analysis
% Choose random points until have at least N of each class
% classes needs to be 1d vector of integer category labels 
%   (label ints don't need to be contiguous)
% EMonson -- 28 Apr 2009

numPts = length(classes);

rr = rand(numPts,1);
[RR,II] = sort(rr);

uniqueClasses = unique(classes);
counts = zeros(length(uniqueClasses),1);

if strcmpi(mode,'duplicates'),
    for ii = 1:numPts,
        randIdx = II(ii);
        Idxs(ii) = randIdx;
        Cats(ii) = classes(randIdx);
        logicalClass = (uniqueClasses == classes(randIdx));
        counts(logicalClass) = counts(logicalClass) + 1;
        if (min(counts) >= N), break; end;
    end;
elseif strcmpi(mode,'number'),
    for ii = 1:N,
        randIdx = II(ii);
        Idxs(ii) = randIdx;
        Cats(ii) = classes(randIdx);
    end;
else
    error(['docChooseRandomPoints: ' mode ' not a valid mode']);
end;
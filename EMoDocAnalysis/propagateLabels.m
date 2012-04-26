function [outArray,outCorrect] = propagateLabels(Idxs, Cats, P, numSteps)
% [labelArray] = propagateLabels(Idxs, Cats, P, numSteps)
% Propagate labels and return "correct" rate over time
% EMonson -- 28 Apr 2009

numIdxs = length(Idxs);
numCats = length(unique(Cats));
numPts = size(P,1);

% Each (numPts x numCats) array leaves out one labeled point for testing
labelBlock = zeros(numPts, numCats, numIdxs);
labelArray = zeros(numPts, numCats);
lastArray = zeros(numPts, numCats);
outArray = zeros(numPts, numCats);
outCorrect = 0;

lastCorrect = 0;

for tt = 1:numSteps,
    
    % Fill known points with ones
    fillBlock();
    fillArray();
    
    % Propagate labels one step
    labelArray = P*labelArray;
    for ss = 1:numIdxs,
        labelBlock(:,:,ss) = P*labelBlock(:,:,ss);
    end;
    
    % "Correctness" check
    correct = 0;
    allZeros = 0;
    ltZeros = 0;
    for ee = 1:numIdxs,
        % Array ee "left out" Idxs(ee) label/source
        [Y,I] = max(labelBlock(Idxs(ee),:,ee),[],2);
        correct = correct + double(I == Cats(ee));
        if (Y == 0), allZeros = allZeros + 1; end;
        if (Y < 0), ltZeros = ltZeros + 1; end;
    end;
    fprintf(1,'Num all 0 tests = %d, all LessThanZero = %d || any allZeroOther %d || ', allZeros, ltZeros, any(sum(lastArray,2)<1e-20));
    correct = correct/numIdxs;
    fprintf(1,'Correct frac = %6.5f\n', correct);
    
    % Quit if further propagation only makes things worse (on average)
    % but keep going if any nodes have all zeros
    if (correct <= lastCorrect && ~any(sum(lastArray,2)<1e-20)), 
        outArray = lastArray;
        outCorrect = lastCorrect;
        break;         
    end;
    
    lastCorrect = correct;
    lastArray = labelArray;
end;

    % Subfunction: Fill ones in labeled spots as "sources" for propagation
    %   leaving one out for each array for testing
    function fillBlock()
        for kk = 1:numIdxs,
            for ii = setdiff(1:numIdxs,kk),     % leave one out
                labelBlock(Idxs(ii),Cats(ii),kk) = 1.0;
                % labelBlock(Idxs(ii),setdiff(1:numCats,Cats(ii)),kk) = -1.0;
            end;
        end;
    end
    % end subfunction fillSources()

    % Subfunction: Fill ones in labeled spots as "sources" for propagation
    %   leaving one out for each array for testing
    function fillArray()
        for ii = 1:numIdxs,
            labelArray(Idxs(ii),Cats(ii)) = 1.0;
            % labelArray(Idxs(ii),setdiff(1:numCats,Cats(ii))) = -1.0;
        end;
    end
    % end subfunction fillSources()
end
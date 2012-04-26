function [outArray,outCorrect,outStd] = propagateLabels3(Idxs, Cats, P, numSteps, OVERLAP, N, M)
% [outArray,outCorrect,outStd] = propagateLabels3(Idxs, Cats, P, numSteps, N, M)
%
% Propagate labels and return "correct" rate over time
% This version changes from "remove 1" to remove N with M replicates
%
% EMonson -- 13 Aug 2009

% OVERLAP = true;

numIdxs = length(Idxs);
numCats = length(unique(Cats));
numPts = size(P,1);

% Non-overlapping below
if (~OVERLAP), M = floor(numIdxs/N); end;

% Each (numPts x numCats) array leaves out one labeled point for testing
labelBlock = zeros(numPts, numCats, M);
labelArray = zeros(numPts, numCats);
lastArray = zeros(numPts, numCats);
outArray = zeros(numPts, numCats);
leaveOuts = zeros(N,M);

correctAvg = zeros(M,1);
correctStd = zeros(M,1);

outCorrect = 0;
outStd = 0;
lastCorrect = 0;
lastStd = 0;

% Reset random number generator so always get same sets with same
% parameters
st = RandStream.getDefaultStream;
reset(st,0);    % using options second argument "seed" just to note it...

% Pick "leave out" indexes from Idxs list (labeled points)
if (~OVERLAP), randPicks = randperm(numIdxs); end;
for jj = 1:M,
    % Overlapping below
    if (OVERLAP), randPicks = randperm(numIdxs); end;
    leaveOuts(:,jj) = randPicks(1:N);
    % Non-overlapping below
    if (~OVERLAP), randPicks = randPicks(N+1:end); end;
end;

% matlabpool('open',8);
for tt = 1:numSteps,
    
    % Fill ones in labeled spots as "sources" for propagation
    %   leaving N out in each array in block for testing
    for kk = 1:M,
        for ii = setdiff(1:numIdxs,leaveOuts(:,kk));     % leave some out
            labelBlock(Idxs(ii),Cats(ii),kk) = 1.0;
            % labelBlock(Idxs(ii),setdiff(1:numCats,Cats(ii)),kk) = -1.0;
        end;
    end;
    % Fill known points with ones (no leave-outs)
    for ii = 1:numIdxs,
        labelArray(Idxs(ii),Cats(ii)) = 1.0;
        % labelArray(Idxs(ii),setdiff(1:numCats,Cats(ii))) = -1.0;
    end;
    
    % Propagate labels one step (no leave-outs)
    labelArray = P*labelArray;

    % Propagate labels for leave-out arrays
    % parfor ss = 1:M,
    for ss = 1:M,
        labelBlock(:,:,ss) = P*labelBlock(:,:,ss);
    end;
   
    % "Correctness" check
    correctArray = zeros(N,M);
    allZeros = 0;
    ltZeros = 0;
    for ee = 1:M,
        % Indexes leaveOuts(:,ee) left out of Idxs label/source
        % max() gives largest label value = assigned category
        [Y,I] = max(labelBlock(Idxs(leaveOuts(:,ee)),:,ee),[],2);
        correctArray(:,ee) = double(I == Cats(leaveOuts(:,ee))');
        allZeros = allZeros + sum(Y==0);
        ltZeros = ltZeros + sum(Y<0);
    end;
     fprintf(1,'Num all 0 tests = %d || any allZeroOther %d || ', allZeros, any(sum(lastArray,2)<1e-20));
    correctArrayMean = mean(correctArray,1);
    for kk = 1:M,
        correctAvg(kk) = mean(correctArrayMean(1:kk),2);  % mean over leaveOuts 1st
        correctStd(kk) = std(correctArrayMean(1:kk),0,2);  % mean over leaveOuts 1st
    end;
     fprintf(1,'Correct frac/std = %6.5f/%6.5f\n', correctAvg(end), correctStd(end));

    figure(10001);
    errorbar(1:M,correctAvg,correctStd./2.0,'k.-');
    hold on;
    plot(1:M,correctStd,'r.-');
    hold off;
    ylim([0 1]);
    drawnow;
    
    % Quit if further propagation only makes things worse (on average)
    % but keep going if any nodes have all zeros
    if (correctAvg(end) <= lastCorrect && ~any(sum(lastArray,2)<1e-20)), 
%    if (correctAvg(end) <= lastCorrect), 
        break;         
    end;
    
    lastCorrect = correctAvg(end);
    lastStd = correctStd(end);
    lastArray = labelArray;
end;

outArray = lastArray;
outCorrect = lastCorrect;
outStd = lastStd;
% matlabpool('close');

end
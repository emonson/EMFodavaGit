clear all;
baseName = 'X20_042709b';

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
fprintf(1,'Loading data set\n');
    load([baseName '.mat']);

% 'difference', 'norm difference', 'filled', 'entropy', 'random',
% 'correlation'
uncertMeasure = 'norm difference';
filledThreshold = 0.8;
scale = 4;
numPrelabeled = 90;

extIdxs = G.Tree{scale,1}.ExtIdxs;
extIdxsSubset = extIdxs(1:numPrelabeled);
cats = classes(:,1);

numIdx = 1;
for num = 6:(numPrelabeled-20),
    if (num > length(extIdxs)), break; end;
    numLog(numIdx) = num;

    fprintf(1,'Choosing scale %d points (%d dups) -- ', scale, num);
    [scaleIdxs,scaleCats] = docChooseScalePoints(classes(:,1),extIdxs,'number',num);

    fprintf(1,'Propagating scale labels\n');
    [scaleLabelArray, optCorrect] = propagateLabels(scaleIdxs,scaleCats,G.P,20);
    [junk,scaleCatOut] = max(scaleLabelArray,[],2);

    scaleCorrect = (scaleCatOut == cats);
    sc(numIdx) = sum(scaleCorrect);

    % Take original points out of both extIdxs and cats
    [notUsedExtIdxsSubset,II] = setdiff(extIdxsSubset,scaleIdxs);

    subsetCorrect = (scaleCatOut(notUsedExtIdxsSubset) == cats(notUsedExtIdxsSubset));
    pc(numIdx) = sum(subsetCorrect)./length(notUsedExtIdxsSubset);
    fprintf(1,'scale correct: %d / %d (%4.3f)\n', sum(scaleCorrect), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));

    numIdx = numIdx + 1;
end;



figure; 
subplot(2,1,1);
plot(numLog,sc,'.-');
title('scale');
subplot(2,1,2);
plot(numLog,pc,'.-');
title([uncertMeasure ' uncertainty measure (mean)']);

clear all;
baseName = 'X20_042709b';

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
fprintf(1,'Loading data set\n');
    load([baseName '.mat']);

scale = 4;
n0cutoff = 0.4;
n1cutoff = 0.18;
n2cutoff = 0.3;
nNcutoff = 0.45;

extIdxs = G.Tree{scale,1}.ExtIdxs;
cats = classes(:,1);

numIdx = 1;
for num = 6:50,
    if (num > length(extIdxs)), break; end;
    numLog(numIdx) = num;

    fprintf(1,'Choosing scale %d points (%d dups) -- ', scale, num);
    [scaleIdxs,scaleCats] = docChooseScalePoints(classes(:,1),extIdxs,'number',num);

    fprintf(1,'Propagating scale labels\n');
    [scaleLabelArray, optCorrect] = propagateLabels(scaleIdxs,scaleCats,G.P,20);
    [junk,scaleCatOut] = max(scaleLabelArray,[],2);

    scaleCorrect = (scaleCatOut == cats);
    sc(numIdx) = sum(scaleCorrect);
    fprintf(1,'scale correct: %d / %d (%4.3f)\n', sum(scaleCorrect), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));

    % Calculate entropy uncertainty and number of source neighbors
    unNbr = zeros([size(scaleLabelArray,1) 1]);
    for jj = 1:length(unNbr),
        % Calculate (incoming) neighbors
        neighborIdxs = setdiff(find(G.P(:,jj)),jj);
        % Find any neighbors which are sources
        neighborSources = intersect(neighborIdxs,scaleIdxs);
        unNbr(jj) = length(neighborSources);
    end;
    normLabels = scaleLabelArray./repmat(sum(scaleLabelArray,2),[1 size(scaleLabelArray,2)]); 
    tmpLog = log10(normLabels);
    tmpLog(isinf(tmpLog)) = 0;
    unEnt = sum( -1.*normLabels.*tmpLog ,2);
    
    % Filter on number of neighbors using entropy cutoffs
    pc(numIdx,1) = sum(unEnt < n0cutoff);
    pc(numIdx,2) = sum(unNbr == 1 & unEnt < n1cutoff);
    pc(numIdx,3) = sum(unNbr == 2 & unEnt < n2cutoff);
    pc(numIdx,4) = sum(unNbr >= 3 & unEnt < nNcutoff);

    numIdx = numIdx + 1;
end;



figure; 
subplot(2,1,1);
plot(numLog,sc,'.-');
title('scale');
subplot(2,1,2);
plot(numLog,pc,'.-');
title('Count w/neighbor and entropy cutoffs');

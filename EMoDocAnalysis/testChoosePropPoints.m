% function [scaleCorrect, scaleCatOut] = testChoosePropPoints(G,classes)
% Test out initial propagation with different "known" point choices

% clear all;
% % baseName = 'X20_042709b';     % Data set used in GUI testing
% baseName = 'X20_061809';
% 
% cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
% fprintf(1,'Loading data set\n');
%     load([baseName '.mat']);
%     % pts = dlmread([baseName '_pts.csv'],' ');

scale = 2;
dups = 3;
% num = 60;
% 'difference', 'norm difference', 'filled', 'entropy', 'random',
% 'diffEnt', 'correlation', 'source neighbor'
uncertMeasure = 'entropy';
filledThreshold = 0.8;

overlap = true;
numReplicates = 30; % Not used if overlap=false

leaveOutFracOfTotal = false;    
numLeaveOut = 3;   % Not used if leaveOutFracOfTotal = true;
leaveOutFrac = 0.2;

% figure; 
% scatter(pts(:,1),pts(:,2),80,classes(:,1),'filled'); 
% title('Real categories');
% colormap(brewerDark1(8));
% caxis([0.5 8.5]);
% colorbar;

fprintf(1,'Choosing scale %d points (%d dups) -- ', scale, dups);
[scaleIdxs,scaleCats] = docChooseScalePoints(classes(:,1),G.Tree{scale,1}.ExtIdxs,'duplicates',dups);
fprintf(1,'%d chosen\n', length(scaleIdxs));

fprintf(1,'Propagating scale labels\n');
if (leaveOutFracOfTotal), numLeaveOut = floor(num*leaveOutFrac); end;
[scaleLabelArray, optCorrect] = propagateLabels3(scaleIdxs,scaleCats,G.P,20,overlap,numLeaveOut,numReplicates);
[junk,scaleCatOut] = max(scaleLabelArray,[],2);
scaleCorrect = (scaleCatOut == classes(:,1));
scaleCorrect(sum(scaleLabelArray,2)<1e-20) = 0;

for ii = 1:length(unique(classes(:,1))),
    C = classes(:,1) == ii;
    S = scaleCatOut == ii;
    TP(ii) = sum(C & S);
    FP(ii) = sum(S & ~C);
    FN(ii) = sum(C & ~S);
    TN(ii) = sum(~C & ~S);
end
precisionMicro = sum(TP)./sum(TP+FP);
recallMicro = sum(TP)./sum(TP+FN);
fprintf(1,'Micro precision = %5.4f, Micro recall = %5.4f\n', precisionMicro, recallMicro);

if strcmpi(uncertMeasure,'norm difference'),
    normLabels = scaleLabelArray./repmat(sum(scaleLabelArray,2),[1 size(scaleLabelArray,2)]);
    normSorted = sort(normLabels,2,'descend');
    uncertainty = 1+diff(normSorted(:,1:2),1,2);
elseif strcmpi(uncertMeasure,'difference'),
    labelsSorted = sort(scaleLabelArray,2,'descend');
    uncertainty = 1+diff(labelsSorted(:,1:2),1,2);
elseif strcmpi(uncertMeasure,'entropy'),
    normLabels = scaleLabelArray./repmat(sum(scaleLabelArray,2),[1 size(scaleLabelArray,2)]); 
    tmpLog = log10(normLabels);
    tmpLog(isinf(tmpLog)) = 0;
    uncertainty = sum( -1.*normLabels.*tmpLog ,2);
elseif strcmpi(uncertMeasure,'diffEnt'),
    labelsSorted = sort(scaleLabelArray,2,'descend');
    normLabels = labelsSorted./repmat(sum(labelsSorted,2),[1 size(labelsSorted,2)]); 
    tmpLog = log10(normLabels);
    tmpLog(isinf(tmpLog)) = 0;
    uncertainty = mean([sum( -1.*normLabels.*tmpLog ,2) (1+diff(normLabels(:,1:2),1,2))],2);
elseif strcmpi(uncertMeasure,'filled'),
    labelsSorted = sort(scaleLabelArray,2,'descend');
    labelsSorted = labelsSorted./repmat(sum(labelsSorted,2),[1 size(labelsSorted,2)]);
    uncertainty = zeros([size(labelsSorted,1) 1]);
    for jj = 1:length(uncertainty),
        % interp1 needs unique values, so adding a little noise before cumsum...
        tmpRow = cumsum( labelsSorted(jj,:) + 0.00001.*rand([1 size(labelsSorted,2)]), 2 );
        uncertainty(jj) = interp1( [0 tmpRow], [0:size(labelsSorted,2)], filledThreshold );
    end;
elseif strcmpi(uncertMeasure,'correlation'),
    normLabelsArray = scaleLabelArray./repmat(sum(scaleLabelArray,2),[1 size(scaleLabelArray,2)]);
    uncertainty = zeros([size(normLabelsArray,1) 1]);
    for jj = 1:length(uncertainty),
        % Calculate (incoming) neighbor correlations
        neighborIdxs = setdiff(find(G.P(:,jj)),jj);
        inNeighborCorrs = corr(normLabelsArray(jj,:)',normLabelsArray(neighborIdxs,:)');
        % Weight them by the incoming edge weights
        inNeighborWeights = G.P(neighborIdxs,jj);
        % But first normalize incoming edge weights
        inNeighborWeights = inNeighborWeights./sum(inNeighborWeights);
        % Because of weighting, mean now calculated by sum()
        uncertainty(jj) = 1-sum(inNeighborCorrs'.*inNeighborWeights);
    end;
elseif strcmpi(uncertMeasure,'source neighbor'),
    uncertainty = zeros([size(normLabelsArray,1) 1]);
    for jj = 1:length(uncertainty),
        % Calculate (incoming) neighbors
        neighborIdxs = setdiff(find(G.P(:,jj)),jj);
        % Find any neighbors which are sources
        neighborSources = intersect(neighborIdxs,scaleIdxs);
        uncertainty(jj) = length(neighborSources);
    end;
elseif strcmpi(uncertMeasure,'weighted source neighbor'),
    uncertainty = zeros([size(normLabelsArray,1) 1]);
    for jj = 1:length(uncertainty),
        % Calculate (incoming) neighbors
        neighborIdxs = setdiff(find(G.P(:,jj)),jj);
        % Find any neighbors which are sources
        neighborSources = intersect(neighborIdxs,scaleIdxs);
        uncertainty(jj) = sum(G.P(neighborSources,jj));
    end;
elseif strcmpi(uncertMeasure,'random'),
    uncertainty = rand([size(scaleLabelArray,1) 1]);
else
    uncertainty = zeros([size(scaleLabelArray,1) 1]);
end;

% Plot uncertainty with correct/incorrect
% figure;
% boxplot(uncertainty,scaleCorrect);
% plot(scaleCorrect,uncertainty,'ro');
% title([uncertMeasure ' uncertainty meaure']);
    
% figure; 
% subplot(2,2,1);
% scatter(pts(:,1),pts(:,2),80,scaleCatOut,'filled'); 
% title('Initial scale point categories');
% colormap(brewerDark1(8));
% caxis([0.5 8.5]);
% 
% subplot(2,2,2); 
% scatter(pts(:,1),pts(:,2),80,scaleCorrect,'filled'); 
% title('Scale point correct(1) & incorrect(0)');
fprintf(1,'scale correct: %d / %d (%4.3f)\n\n', sum(scaleCorrect), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));

fprintf(1,'Choosing random points (%d dups) -- ', dups);
[randIdxs,randCats] = docChooseRandomPoints(classes(:,1),'duplicates',dups);
% [randIdxs,randCats] = docChooseRandomPoints(classes(:,1),'number',length(scaleIdxs));
% [randIdxs,randCats] = docChooseRandomPoints(classes(:,1),'number',num);
fprintf(1,'%d chosen\n', length(randIdxs));

fprintf(1,'Propagating random labels\n');
randLabelArray = propagateLabels(randIdxs,randCats,G.P,20);
[junk,randCatOut] = max(randLabelArray,[],2);
fprintf(1,'Number all zero propagated results: %d\n', length(find(junk==0.0)) );

if strcmpi(uncertMeasure,'entropy'),
    normRLabels = randLabelArray./repmat(sum(randLabelArray,2),[1 size(randLabelArray,2)]); 
    tmpRLog = log10(normRLabels);
    tmpRLog(isinf(tmpRLog)) = 0;
    uncertaintyR = sum( -1.*normRLabels.*tmpRLog ,2);
end;
% 
% % subplot(2,2,3);
% % scatter(pts(:,1),pts(:,2),80,randCatOut,'filled'); 
% % title('Initial random point categories');
% % colormap(brewerDark1(8));
% % caxis([0.5 8.5]);
% % 
% % subplot(2,2,4); 
randCorrect = (randCatOut == classes(:,1));
randCorrect(sum(randLabelArray,2)<1e-20) = 0;
% scatter(pts(:,1),pts(:,2),80,randCorrect,'filled'); 
% title('Random point correct(1) & incorrect(0)');
fprintf(1,'random correct: %d / %d (%4.3f)\n\n', sum(randCorrect), length(randCorrect), sum(randCorrect)/length(randCorrect));

% Rescale weights
% normLabelArray = scaleLabelArray./repmat(sum(scaleLabelArray,2),1,size(scaleLabelArray,2));
% fprintf(1,'Calculating label correlations\n');
% normLabelCorr = corr(normLabelArray');
% 
% nG.W = G.W.* ((((normLabelCorr)+1)./2).^4);
% nG = ConstructGraphOperators(nG);
% 
% fprintf(1,'Propagating scale labels on new graph\n');
%     newScaleLabelArray = propagateLabels(scaleIdxs,scaleCats,nG.P,200);
%     [junk,newScaleCatOut] = max(newScaleLabelArray,[],2);
% 
% figure; 
% subplot(1,3,1);
% scatter(pts(:,1),pts(:,2),80,newScaleCatOut,'filled'); 
% title('Rescaled categories (scale init)');
% colormap(brewerDark1(8));
% caxis([0.5 8.5]);
% 
% subplot(1,3,2); 
% newScaleCorrect = (newScaleCatOut == classes(:,1));
% scatter(pts(:,1),pts(:,2),80,newScaleCorrect,'filled'); 
% title('Rescaled correct/incorrect');
% fprintf(1,'new scale correct: %d\n', sum(newScaleCorrect));
% 
% subplot(1,3,3); 
% newScaleDiff = (newScaleCorrect - scaleCorrect);
% scatter(pts(:,1),pts(:,2),80,newScaleDiff,'filled'); 
% title('Rescaled gain/loss');
% caxis([-5 4]);
% fprintf(1,'Added: %d, Lost: %d\n', sum(newScaleDiff>0), sum(newScaleDiff<0));
% 
% % Calculate number of mixed labels / uncertainty in labels
% numCats = size(newScaleLabelArray,2);
% scaleLabelNorm = sort(newScaleLabelArray,2)./repmat(sum(newScaleLabelArray,2),1,numCats);
% % [junk,snI] = max(diff(scaleLabelNorm,1,2),[],2);
% % scaleLabelNumMixed = numCats - snI;
% scaleLabelNumMixed = sum(scaleLabelNorm>0.10,2);
% figure; 
% subplot(1,3,1);
% scatter(newPts(:,1),newPts(:,2),80,classes(:,1),'filled'); 
% title('Real classes on rescaled graph');
% colormap(brewerDark1(8)); 
% caxis([0.5 8.5]);
% % colorbar;
% 
% subplot(1,3,2);
% scatter(newPts(:,1),newPts(:,2),80,9-scaleLabelNumMixed,'filled'); 
% title('Number of dominant mixed colors (reverse color map)');
% caxis([0.5 8.5]);
% 
% subplot(1,3,3);
% scatter(newPts(:,1),newPts(:,2),80,newScaleCatOut,'filled'); 
% title('Rescaled propagated classes on rescaled graph');
% caxis([0.5 8.5]);
% % colorbar;

% Rescale weights based this time on thresholded normalized color labels
% fprintf(1,'Calculating label correlations\n');
% normThreshLabelCorr = corr((normLabelArray>0.10)');
% 
% ntG.W = G.W.* ((((normThreshLabelCorr)+1)./2).^4);
% ntG = ConstructGraphOperators(ntG);
% 
% fprintf(1,'Propagating scale labels on new graph\n');
%     newThreshScaleLabelArray = propagateLabels(scaleIdxs,scaleCats,ntG.P,200);
%     [junk,newThreshScaleCatOut] = max(newThreshScaleLabelArray,[],2);
% 
% figure; 
% subplot(1,3,1);
% scatter(newPts(:,1),newPts(:,2),80,newThreshScaleCatOut,'filled'); 
% title('Rescaled Thresh categories (scale init)');
% colormap(brewerDark1(8));
% caxis([0.5 8.5]);
% 
% subplot(1,3,2); 
% newThreshScaleCorrect = (newThreshScaleCatOut == classes(:,1));
% scatter(newPts(:,1),newPts(:,2),80,newThreshScaleCorrect,'filled'); 
% title('Rescaled Thresh correct/incorrect');
% fprintf(1,'new scale correct: %d\n', sum(newThreshScaleCorrect));
% 
% subplot(1,3,3); 
% newThreshScaleDiff = (newThreshScaleCorrect - scaleCorrect);
% scatter(newPts(:,1),newPts(:,2),80,newThreshScaleDiff,'filled'); 
% title('Rescaled Thresh gain/loss');
% caxis([-5 4]);
% fprintf(1,'Added: %d, Lost: %d\n', sum(newThreshScaleDiff>0), sum(newThreshScaleDiff<0));


% end

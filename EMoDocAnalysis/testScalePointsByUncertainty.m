% clear all;
% baseName = 'X20_042709b';
% 
% cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
% fprintf(1,'Loading data set\n');
% load([baseName '.mat']);

scale = 4;
initNum = 25;       % Original number of points picked
incNum = 1;     % How many each time when picking uncertain points
maxRuns = 60;
% 'difference', 'norm difference', 'filled', 'entropy', 'random',
% 'correlation'
uncertMeasure = 'entropy';
filledThreshold = 0.8;

extIdxs = G.Tree{scale,1}.ExtIdxs;
cats = classes(:,1);

runs = floor((length(extIdxs)-initNum)/incNum);
if (runs > maxRuns), runs = maxRuns; end;

scaleIdxs = [];
scaleCats = [];
sc = zeros([runs 1]);
ptCount = zeros([runs 1]);
un = zeros([runs 1]);
gc = zeros([runs 1]);
prevGuesses = zeros(size(classes(:,1)));

for ii = 1:runs,
    
    if (ii==1), num = initNum;
    else num = incNum;
    end;
    
    % Pick points from list of multiscale indices
    % fprintf(1,'Choosing scale %d points (num: %d) -- ', scale, num);
    [pickedIdxs,pickedCats] = docChooseScalePoints(classes(:,1),extIdxs,'number',num);
    
    % Append newly picked points onto old list
    scaleIdxs = cat(2,scaleIdxs,pickedIdxs);
    scaleCats = cat(2,scaleCats,pickedCats);
    
    % Propagate points
    % fprintf(1,'Propagating scale labels\n');
    [scaleLabelArray,outCorrect] = propagateLabels(scaleIdxs,scaleCats,G.P,20);
    
    % Find which are really correct
    [junk,scaleCatOut] = max(scaleLabelArray,[],2);
    scaleCorrect = (scaleCatOut == classes(:,1));
    % Record original number correct
    sc(ii) = sum(scaleCorrect);
    if (ii==1), ptCount(ii) = initNum;
    else ptCount(ii) = ptCount(ii-1) + incNum;
    end;
    fprintf(1,'\nPts: %d || scale correct: %d / %d (%4.3f)\n\n', ptCount(ii), sc(ii), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));

    % Take original points out of both extIdxs and cats
    [extIdxs,II] = setdiff(extIdxs,pickedIdxs);

    if strcmpi(uncertMeasure,'norm difference'),
        labelsSorted = sort(scaleLabelArray,2,'descend');
        labelsSorted = labelsSorted./repmat(sum(labelsSorted,2),[1 size(labelsSorted,2)]);
        uncertainty = 1+diff(labelsSorted(:,1:2),1,2);
    elseif strcmpi(uncertMeasure,'difference'),
        labelsSorted = sort(scaleLabelArray,2,'descend');
        uncertainty = 1+diff(labelsSorted(:,1:2),1,2);
    elseif strcmpi(uncertMeasure,'entropy'),
        normLabels = scaleLabelArray./repmat(sum(scaleLabelArray,2),[1 size(scaleLabelArray,2)]); 
        tmpLog = log10(normLabels);
        tmpLog(isinf(tmpLog)) = 0;
        uncertainty = sum( -1.*normLabels.*tmpLog ,2);
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
    elseif strcmpi(uncertMeasure,'random'),
        uncertainty = rand([size(scaleLabelArray,1) 1]);
    else
        uncertainty = zeros([size(scaleLabelArray,1) 1]);
    end;
    
    % un(ii) = mean(uncertainty);
    un(ii) = outCorrect;
    % Keep only uncertainty values for remaining extIdxs points
    extUncert = uncertainty(extIdxs);
    % Sort these
    [junk,II] = sort(extUncert,1,'descend');
    % Reorder remaining extIdxs and cats according to uncertainty
    extIdxs = extIdxs(II);

    gc(ii) = sum(prevGuesses ~= scaleCatOut);
    prevGuesses = scaleCatOut;

end;

figure; 
subplot(3,1,1);
plot(ptCount,sc,'.-');
line([20 100],[700 780],'Color','k');
% line([150 450],[1500 1800],'Color','k');
axis([20 90 700 805]);
titleStr = sprintf('scale %d, initNum %d, inc %d, %s uncert',scale,initNum,incNum,uncertMeasure);
title(titleStr);
subplot(3,1,2);
plot(ptCount,un,'r.-');
% axis([20 90 0.2 0.8]);
subplot(3,1,3);
gcTmp = gc;
gcTmp(1) = 0;
diffTmp = diff(un);
gcTmp(2:end) = gcTmp(2:end).*(-1.*(diffTmp<-0.01)+(diffTmp>=-0.01)); % put negatives back in
gcTmp = gcTmp./4.0;   % Scaling factor
plot(ptCount,gcTmp,'.-','Color',[0.5 0.5 0.5]);
hold on;
plot(ptCount(2:end),diff(sc),'b.-');
maxVal = 1.1*max(abs(cat(1,diff(sc),gcTmp)));
% axis([20 90 -maxVal maxVal]);


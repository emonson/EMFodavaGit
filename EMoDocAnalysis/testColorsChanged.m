clear all;
baseName = 'X20_042709b';

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
fprintf(1,'Loading data set\n');
    load([baseName '.mat']);

% 'difference', 'norm difference', 'filled', 'entropy', 'random',
% 'correlation'
uncertMeasure = 'entropy';
filledThreshold = 0.8;

scale = 4;
extIdxs = G.Tree{scale,1}.ExtIdxs;
prevGuesses = zeros(size(classes(:,1)));
% Test randomizing extIdxs order
% extIdxs = extIdxs(randperm(length(extIdxs)));
numIdx = 1;
for num = 6:1:50,
    if (num > length(extIdxs)), break; end;
    numLog(numIdx) = num;

    fprintf(1,'Choosing scale %d points (%d dups) -- ', scale, num);
        [scaleIdxs,scaleCats] = docChooseScalePoints(classes(:,1),extIdxs,'number',num);
        % [scaleIdxs,scaleCats] = docChooseRandomPoints(classes(:,1),'number',num);
    fprintf(1,'Propagating scale labels\n');
        [scaleLabelArray, optCorrect] = propagateLabels(scaleIdxs,scaleCats,G.P,20);
        [junk,scaleCatOut] = max(scaleLabelArray,[],2);

    scaleCorrect = (scaleCatOut == classes(:,1));
    sc(numIdx) = sum(scaleCorrect);

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

    % pc(numIdx,scaleIdx) = mean(uncertainty);
    pc(numIdx) = mean(uncertainty);
    fprintf(1,'scale correct: %d / %d (%4.3f)\n', sum(scaleCorrect), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));

    gc(numIdx) = sum(prevGuesses ~= scaleCatOut);
    prevGuesses = scaleCatOut;
    numIdx = numIdx + 1;
end;

figure; 
subplot(3,1,1);
plot(numLog,sc,'k.-');
title('scale');
subplot(3,1,2);
plot(numLog,pc,'r.-');
title([uncertMeasure ' uncertainty measure (mean)']);
subplot(3,1,3);
gcTmp = gc;
gcTmp(1) = 0;
diffTmp = diff(pc);
gcTmp(2:end) = gcTmp(2:end).*(-1.*(diffTmp<-0.05)+(diffTmp>=-0.05)); % put negatives back in
gcTmp = gcTmp./3;   % Scaling factor
plot(numLog,gcTmp,'.-','Color',[0.5 0.5 0.5]);
hold on;
plot(numLog(2:end),diff(sc),'k.-');
maxVal = 1.1*max(abs(cat(2,diff(sc),gcTmp)));


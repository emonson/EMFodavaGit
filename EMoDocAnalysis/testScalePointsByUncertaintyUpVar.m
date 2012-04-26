clear all;
baseName = 'X20_042709b';

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
fprintf(1,'Loading data set\n');
load([baseName '.mat']);

scale = 4;
initNum = 25;       % Original number of points picked
incNum = 1;     % How many each time when picking uncertain points
allowedDrop = 0;    % How far overall accuracy allowed to drop before points are thrown out.
removeAmt = 'one';  % 'set' or 'one' for removing if see large drop
maxRuns = 65;
reorderByUncertainty = 0;   % If ~= 1, then using natural order
% 'difference', 'norm difference', 'filled', 'entropy', 'random',
% 'correlation'
uncertMeasure = 'entropy';
filledThreshold = 0.8;

extIdxs = G.Tree{scale,1}.ExtIdxs;
runs = floor((length(extIdxs)-initNum)/incNum);
if (runs > maxRuns), runs = maxRuns; end;
sc = zeros([runs 1]);
ptCount = zeros([runs 1]);
un = zeros([runs 1]);
gc = zeros([runs 1]);

varIdx = 1; % indexing for any variations

for allowedDrop = [100 10 5 3 2 1 0],
    varVal{varIdx} = int2str(allowedDrop);

    extIdxs = G.Tree{scale,1}.ExtIdxs;

    scaleIdxs = [];
    scaleCats = [];
    labeledIdxs = [];
    exitAll = 0;
    prevGuesses = zeros(size(classes(:,1)));

    for ii = 1:runs,

        if (ii==1), num = initNum;
        else num = incNum;
        end;

        done = 0;

        extIdxsTmp = extIdxs;
        deleteCount = 0;
        while (done == 0),
            % Pick points from list of multiscale indices
            % fprintf(1,'Choosing scale %d points (num: %d) -- ', scale, num);
            try
                [pickedIdxs,pickedCats] = docChooseScalePoints(classes(:,1),extIdxsTmp,'number',num);
            catch ME1
                fprintf(1,'picking error');
                exitAll = 1;
                break;
            end;

            % Append newly picked points onto old list
            scaleIdxs = cat(2,scaleIdxs,pickedIdxs);
            scaleCats = cat(2,scaleCats,pickedCats);
            if(ii==1),
                labeledIdxsTmp = scaleIdxs;
            else
                % Want to keep track of which idxs have been labeled in earlier
                % rounds, so if they get reused, we don't count them again
                labeledIdxsTmp = union(labeledIdxsTmp,pickedIdxs);
            end;

            % Propagate points
            % fprintf(1,'Propagating scale labels\n');
            scaleLabelArray = propagateLabels(scaleIdxs,scaleCats,G.P,20);

            % Find which are really correct
            [junk,scaleCatOut] = max(scaleLabelArray,[],2);
            scaleCorrect = (scaleCatOut == classes(:,1));
            if((ii > 1) && (sum(scaleCorrect) > sc(ii-1,varIdx)-allowedDrop)),
                fprintf(1,'%d used but deleted\n', deleteCount);
                done = 1;
            elseif (ii == 1),
                done = 1;
            else
                % If recent answer not better, remove those points from lists
                % and redo propagation
                if(strcmpi(removeAmt,'set')), delNum = num;
                else delNum = 1;
                end; 
                deleteCount = deleteCount + delNum;
                scaleIdxs = scaleIdxs(1:end-delNum);
                scaleCats = scaleCats(1:end-delNum);
                extIdxsTmp = extIdxsTmp(delNum+1:end);
            end;
        end;
        if(exitAll), 
            % Ran out of points, so get rid of unused array slots
            sc(ii:end,varIdx) = sc(ii-1,varIdx);
            ptCount(ii:end,varIdx) = ptCount(ii-1,varIdx);
            gc(ii:end,varIdx) = gc(ii-1,varIdx);
            un(ii:end,varIdx) = un(ii-1,varIdx);
            break; 
        end;
        newIdxs = setdiff(labeledIdxsTmp,labeledIdxs);
        labeledIdxs = labeledIdxsTmp;
        fprintf(1,'\nscale correct: %d / %d (%4.3f)\n', sum(scaleCorrect), length(scaleCorrect), sum(scaleCorrect)/length(scaleCorrect));
        % Record original number correct
        sc(ii,varIdx) = sum(scaleCorrect);
        if (ii==1), ptCount(ii,varIdx) = initNum;
        else ptCount(ii,varIdx) = ptCount(ii-1,varIdx) + length(newIdxs);
        end;

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

        un(ii,varIdx) = mean(uncertainty);
        % Keep only uncertainty values for remaining extIdxs points
        extUncert = uncertainty(extIdxs);
        % Sort these
        [junk,II] = sort(extUncert,1,'descend');
        % Reorder remaining extIdxs and cats according to uncertainty
        if(reorderByUncertainty == 1),
            extIdxs = extIdxs(II);
        end;

        gc(ii,varIdx) = sum(prevGuesses ~= scaleCatOut);
        prevGuesses = scaleCatOut;

    end;
    
    varIdx = varIdx + 1;
end; % end variations

figure; 
plot(ptCount,sc,'.-');
line([20 100],[700 780],'Color','k');
axis([20 90 700 805]);
titleStr = sprintf('scale %d, initNum %d, inc %d, %s removal, %s uncert',scale,initNum,incNum,removeAmt,uncertMeasure);
title(titleStr);
legend(varVal);

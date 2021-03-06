%% Using glmnet to compute linear regression with lasso (or elastic net)
% constraints

INCLUDE_DELTA_IN_MS = true;

% If we are solving for a document classification problem, 
% Y is the response vector of assigned labels w/dim [Ndocs 1]
% X is the predictor and has dim [Ndocs Mterms] or similar

% Change set based on direction of term-document matrix
% X = full(tdm_norm');
% Y = double(classes(:,1));
X = full(tdm_norm);
Y = double(classes(:,1));

s = RandStream.create('mt19937ar','seed',1000);
RandStream.setDefaultStream(s);

%% Remove 10% subset for cross-validation of fit

numSubsets = 10;
nlambda = 100;  % default

linMeanSqError = zeros(1, numSubsets);
fitMeanSqError = zeros(nlambda, numSubsets);
fitMsMeanSqError = zeros(nlambda, numSubsets);
fitMeanSqErrorTrain = zeros(nlambda, numSubsets);
fitMsMeanSqErrorTrain = zeros(nlambda, numSubsets);

fitArray = [];
fitMsArray = [];
setArray = [];

%% Try using term-term scaling functions as basis for fit
% after projecting document vectors onto these. 

if(INCLUDE_DELTA_IN_MS)
    % Delta basis is all ones 
    % Initialize basis function group and record of which scales fcns are from
    extBases = eye(size(X,2));
    scales = zeros(1,size(X,2));
else
    % Use these if not adding in original (delta) doc vectors as basis functions
    extBases = [];
    scales = [];
end

%% Gather up multi-scale basis functions from G.Tree structure
fprintf(1,'Gathering ExtBasis functions\n');
for ii = 1:size(G.Tree,1),
    basisTmp = G.Tree{ii,1}.ExtBasis;
    % Smooth the mean values of the basis function to get a reasonable (not noisy)
    %   cutoff for "good" basis functions (mean over certain threshold)
    smMean = smooth(full(mean(G.Tree{ii,1}.ExtBasis,1)),0.05,'lowess');
    % Grab the first point (index) that goes under a certain threshold
    idxLimit = find(smMean < 0.0001, 1);
    extBases = cat(2,extBases,basisTmp(:,1:idxLimit));
    
    scaleTmp = ii*ones(1,idxLimit);
    scales = cat(2,scales,scaleTmp);
end;

%% Do projection of doc vectors onto new basis and then fit

fprintf(1,'Doing projection of term freq onto MS basis\n');
Xms = X*extBases;

%% Set options for glmnet fit

options = glmnetSet;
options.type = 'naive';
options.nlambda = nlambda;
options.alpha = 0.9;

%% Doing fits on full data sets for later use

fprintf(1,'Computing Lasso fit on full data set\n');
fit_full = glmnet(X, Y, 'gaussian', options);
fprintf(1,'Computing Lasso on full MS bases set\n');
fitMS_full = glmnet(Xms, Y, 'gaussian', options);
figure(3021);
stem(find(fit_full.beta(:,100)),fit_full.beta(logical(fit_full.beta(:,100)),100),'ko','Color',[0 0 0.8],'MarkerSize',8);
title('delta fit beta 100s');
hold on;
figure(3022);
stem(find(fitMS_full.beta(:,100)),fitMS_full.beta(logical(fitMS_full.beta(:,100)),100),'ko','Color',[0 0 0.8],'MarkerSize',8);
title('MS fit beta 100s');
hold on;

pp = randperm(size(X,1));
pp_sets = mod(0:(size(X,1)-1), numSubsets);

%% Main loop over test/train sets
%
for ss = 1:numSubsets,
    
    fprintf(1,'\n# # Iteration %d\n\n', ss);
    ppSub = pp(pp_sets == (ss-1));  % mod output starts at 0
    [SS,II] = sort(Y(ppSub));
    testSet = ppSub(II);
    trainSet = setdiff(1:size(X,1),testSet);
    
    setArray(ss).testSet = testSet;
    setArray(ss).trainSet = trainSet;

    %% Compute Lasso fit to training split
    
    fprintf(1,'Computing Lasso fit\n');
    fit = glmnet(X(trainSet,:),Y(trainSet),'gaussian',options);
    figure(3021);
    stem(find(fit.beta(:,100)),fit.beta(logical(fit.beta(:,100)),100),'k.','Color',[ss*(1.0/numSubsets) 0 0]);
    drawnow;
    % fitArray(ss) = fit;

    %% Also compute straight linear regression with matlab
    
    fprintf(1,'Computing Least Squares fit\n');
    beta = [ones(length(trainSet),1) X(trainSet,:)]\Y(trainSet);

    %% Then fit on to MS basis projection
    
    fprintf(1,'Computing Lasso on MS bases\n');
    fitMS = glmnet(Xms(trainSet,:),Y(trainSet),'gaussian',options);
    figure(3022);
    stem(find(fitMS.beta(:,100)),fitMS.beta(logical(fitMS.beta(:,100)),100),'k.','Color',[ss*(1.0/numSubsets) 0 0]);
    drawnow;
    % fitMsArray(ss) = fitMS;

    %% Calculate errors on remaining test section of data set for all values
    % of contraint (shrinkage) parameter lambda
    
    fprintf(1,'Calculating fit errors\n');
    for ii = 1:size(fit.beta,2),
        fitMeanSqError(ii,ss) = mean((Y(testSet)-(X(testSet,:)*fit.beta(:,ii)+fit.a0(ii))).^2);
        fitMsMeanSqError(ii,ss) = mean((Y(testSet)-(Xms(testSet,:)*fitMS.beta(:,ii)+fitMS.a0(ii))).^2);
        fitMeanSqErrorTrain(ii,ss) = mean((Y(trainSet)-(X(trainSet,:)*fit.beta(:,ii)+fit.a0(ii))).^2);
        fitMsMeanSqErrorTrain(ii,ss) = mean((Y(trainSet)-(Xms(trainSet,:)*fitMS.beta(:,ii)+fitMS.a0(ii))).^2);
    end
    linMeanSqError(ss) = mean((Y(testSet)-[ones(length(testSet),1) X(testSet,:)]*beta).^2);
    
    %% Pring error summaries
    
    fprintf(1,'\nMean squared error\n');
    fprintf(1,'Least squares: \t\t%3.2f \t%d/%d terms\n', mean((Y(testSet)-[ones(length(testSet),1) X(testSet,:)]*beta).^2), sum(beta~=0), length(beta) );
    fprintf(1,'Lasso terms: \t\t%3.2f \t%d/%d terms\n', mean((Y(testSet)-(X(testSet,:)*fit.beta(:,end)+fit.a0(end))).^2), fit.df(end), size(fit.beta,1) );
    fprintf(1,'Lasso MS end: \t\t%3.2f \t%d/%d functions\n', mean((Y(testSet)-(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end))).^2), fitMS.df(end), size(fitMS.beta,1) );
    fprintf(1,'Lasso MS end*0.8: \t%3.2f \t%d/%d functions\n', mean((Y(testSet)-(Xms(testSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8))).^2), fitMS.df(end*0.8), size(fitMS.beta,1) );
    fprintf(1,'Lasso MS end*0.6: \t%3.2f \t%d/%d functions\n', mean((Y(testSet)-(Xms(testSet,:)*fitMS.beta(:,end*0.6)+fitMS.a0(end*0.6))).^2), fitMS.df(end*0.6), size(fitMS.beta,1) );

    fprintf(1,'\nSqrt mean squared error after rounding\n');
    fprintf(1,'Least squares: \t\t%3.2f \t%d/%d terms\n', sqrt(mean((Y(testSet)-round([ones(length(testSet),1) X(testSet,:)]*beta)).^2)), sum(beta~=0), length(beta) );
    fprintf(1,'Lasso terms: \t\t%3.2f \t%d/%d terms\n', sqrt(mean((Y(testSet)-round(X(testSet,:)*fit.beta(:,end)+fit.a0(end))).^2)), fit.df(end), size(fit.beta,1) );
    fprintf(1,'Lasso MS end: \t\t%3.2f \t%d/%d functions\n', sqrt(mean((Y(testSet)-round(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end))).^2)), fitMS.df(end), size(fitMS.beta,1) );
    fprintf(1,'Lasso MS end*0.8: \t%3.2f \t%d/%d functions\n', sqrt(mean((Y(testSet)-round(Xms(testSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8))).^2)), fitMS.df(end*0.8), size(fitMS.beta,1) );
    fprintf(1,'Lasso MS end*0.6: \t%3.2f \t%d/%d functions\n', sqrt(mean((Y(testSet)-round(Xms(testSet,:)*fitMS.beta(:,end*0.6)+fitMS.a0(end*0.6))).^2)), fitMS.df(end*0.6), size(fitMS.beta,1) );

    %% Plot predictions on test data section
    
%     figure; 
%     plot(Y(testSet),'k.','Color',[0.9 0.9 0.9]); 
%     hold on; 
%     plot(X(testSet,:)*fit.beta(:,end)+fit.a0(end),'bo'); 
%     plot([ones(length(testSet),1) X(testSet,:)]*beta,'mo');
%     % plot(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'bs');
%     plot(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'rs');
%     % plot(Xms(testSet,:)*fitMS.beta(:,end*0.5)+fitMS.a0(end*0.5),'gs');
%     % ylim([-1 10]);
%     titleStr = sprintf('Iteration %d', ss);
%     title(titleStr);

    %% Viewing fit to original training section of data set
    
%     figure; 
%     plot(Y(trainSet),'k.','Color',[0.9 0.9 0.9]); 
%     hold on; 
%     plot(X(trainSet,:)*fit.beta(:,end)+fit.a0(end),'bo'); 
%     plot([ones(length(trainSet),1) X(trainSet,:)]*beta,'mo');
%     plot(Xms(trainSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'rs');
    % plot(Xms(trainSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8),'rs');
    % plot(Xms(trainSet,:)*fitMS.beta(:,end*0.6)+fitMS.a0(end*0.6),'gs');
    % % ylim([-1 10]);


end


%% Round values for assignments to categories

% figure; 
% plot(Y(testSet),'k.'); 
% hold on; 
% plot(round(X(testSet,:)*fit.beta(:,end)+fit.a0(end)),'bo'); 
% plot(round([ones(length(testSet),1) X(testSet,:)]*beta),'mo');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end)),'bs');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end*0.75)+fitMS.a0(end)),'rs');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end*0.5)+fitMS.a0(end)),'gs');

%% Viewing fit to original training section of data set

% figure; 
% plot(Y(trainSet),'k.'); 
% hold on; 
% plot(X(trainSet,:)*fit.beta(:,end)+fit.a0(end),'bo'); 
% plot([ones(length(trainSet),1) X(trainSet,:)]*beta,'mo');
% plot(Xms(trainSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'bs');
% plot(Xms(trainSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8),'rs');
% plot(Xms(trainSet,:)*fitMS.beta(:,end*0.6)+fitMS.a0(end*0.6),'gs');
% % ylim([-1 10]);

%% Quantile plot of original scales vs scales remaining after fit selection

figure; 
quantileplot(scales,'k.-');
hold on;
quantileplot(scales(fitMS_full.beta(:,end)~=0),'rs-');
% quantileplot(scales(fitMS.beta(:,end*0.8)~=0),'go-');
quantileplot(scales(fitMS.beta(:,end*0.6)~=0),'gs-');

%% Plot to look for minimum error vs lamba fit parameter

figure;
plot(fitMeanSqError, 'bo-');
hold on;
plot(fitMsMeanSqError, 'rs-');
title('Testing set');

%% Plot to look for minimum error vs lamba fit parameter

figure;
errorbar(mean(fitMeanSqError,2),std(fitMeanSqError,0,2), 'bo-');
hold on;
errorbar(mean(fitMsMeanSqError,2),std(fitMsMeanSqError,0,2), 'rs-');
title('Testing set');

%% Plot to look at how models fit the training data

% figure;
% plot(fitMeanSqErrorTrain, 'bo-');
% hold on;
% plot(fitMsMeanSqErrorTrain, 'rs-');
% title('Training set');


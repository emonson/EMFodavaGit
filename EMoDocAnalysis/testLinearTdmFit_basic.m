% Using glmnet to compute linear regression with lasso (or elastic net)
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

% Remove 10% subset for cross-validation of fit
pp = randperm(size(X,1));
ppSub = pp(3*floor(size(X,1)/10)+(1:floor(size(X,1)/10)));
[SS,II] = sort(Y(ppSub));
testSet = ppSub(II);
trainSet = setdiff(1:size(X,1),testSet);

% Set options for glmnet fit
options = glmnetSet;
options.type = 'naive';
options.alpha = 0.5;
options.nlambda = 100;
fprintf(1,'Computing Lasso fit\n');
fit = glmnet(X(trainSet,:),Y(trainSet),'gaussian',options);

% Also compute straight linear regression with matlab
fprintf(1,'Computing Least Squares fit\n');
beta = [ones(length(trainSet),1) X(trainSet,:)]\Y(trainSet);

% Try using term-term scaling functions as basis for fit
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

% Gather up multi-scale basis functions from G.Tree structure
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

% Do projection of doc vectors onto new basis and then fit
fprintf(1,'Projecting term frequencies onto MS basis\n');
Xms = X*extBases;
fprintf(1,'Computing Lasso on MS bases\n');
fitMS = glmnet(Xms(trainSet,:),Y(trainSet),'gaussian',options);

% Calculate errors on remaining test section of data set for all values
% of contraint (shrinkage) parameter lambda
fprintf(1,'Calculating fit errors\n');
fitMeanSqError = zeros(size(fit.beta,2),1);
for ii = 1:size(fit.beta,2),
    fitMeanSqError(ii) = mean((Y(testSet)-(X(testSet,:)*fit.beta(:,ii)+fit.a0(ii))).^2);
end
fitMsMeanSqError = zeros(size(fitMS.beta,2),1);
for ii = 1:size(fit.beta,2),
    fitMsMeanSqError(ii) = mean((Y(testSet)-(Xms(testSet,:)*fitMS.beta(:,ii)+fitMS.a0(ii))).^2);
end

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

% Plot predictions on test data section
figure; 
plot(Y(testSet),'k.','Color',[0.9 0.9 0.9]); 
hold on; 
plot(X(testSet,:)*fit.beta(:,end)+fit.a0(end),'bo'); 
% plot([ones(length(testSet),1) X(testSet,:)]*beta,'k+');
% plot(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'bs');
% plot(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'rs');
plot(Xms(testSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8),'rs');
% ylim([-1 10]);

% Round values for assignments to categories
% figure; 
% plot(Y(testSet),'k.'); 
% hold on; 
% plot(round(X(testSet,:)*fit.beta(:,end)+fit.a0(end)),'bo'); 
% plot(round([ones(length(testSet),1) X(testSet,:)]*beta),'mo');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end)+fitMS.a0(end)),'bs');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end*0.75)+fitMS.a0(end)),'rs');
% plot(round(Xms(testSet,:)*fitMS.beta(:,end*0.5)+fitMS.a0(end)),'gs');

% Viewing fit to original training section of data set
% figure; 
% plot(Y(trainSet),'k.'); 
% hold on; 
% plot(X(trainSet,:)*fit.beta(:,end)+fit.a0(end),'bo'); 
% plot([ones(length(trainSet),1) X(trainSet,:)]*beta,'mo');
% plot(Xms(trainSet,:)*fitMS.beta(:,end)+fitMS.a0(end),'bs');
% plot(Xms(trainSet,:)*fitMS.beta(:,end*0.8)+fitMS.a0(end*0.8),'rs');
% plot(Xms(trainSet,:)*fitMS.beta(:,end*0.6)+fitMS.a0(end*0.6),'gs');
% % ylim([-1 10]);

% Quantile plot of original scales vs scales remaining after fit selection
figure; 
quantileplot(scales,'k.-');
hold on;
quantileplot(scales(fitMS.beta(:,end)~=0),'rs-');
quantileplot(scales(fitMS.beta(:,end*0.8)~=0),'gs-');
% quantileplot(scales(fitMS.beta(:,end*0.6)~=0),'ys-');

% Plot to look for minimum error vs lamba fit parameter
figure;
plot(fitMeanSqError,'bo-');
hold on;
plot(fitMsMeanSqError,'rs-');


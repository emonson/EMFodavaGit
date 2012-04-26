correctSort = sort(scaleLabelArray(scaleCorrect,:),2,'descend');
incorrectSort = sort(scaleLabelArray(~scaleCorrect,:),2,'descend');

correctSort = correctSort./repmat(sum(correctSort,2),[1 8]);
incorrectSort = incorrectSort./repmat(sum(incorrectSort,2),[1 8]);

figure; 
subplot(1,2,1); 
    boxplot(correctSort,'colors','b','symbol','b+');
    axis([0.5 8.5 -0.04 1]); 
    title('Correct');
subplot(1,2,2); 
    boxplot(incorrectSort,'colors','k','symbol','ko');
    axis([0.5 8.5 -0.04 1]); 
    title('Incorrect');

figure; 
subplot(1,2,1);
    plot(correctSort','k'); axis([0.5 8.5 0 1]); title('Correct'); 
subplot(1,2,2); 
    plot(incorrectSort','k'); axis([0.5 8.5 0 1]); title('Incorrect');

figure; 
subplot(1,2,1);
    boxplot(diff(correctSort,1,2),'colors','b','symbol','b+'); axis([0.5 8.5 -1 0.04]); title('Correct diff');
subplot(1,2,2); 
    boxplot(diff(incorrectSort,1,2),'colors','k','symbol','ko'); axis([0.5 8.5 -1 0.04]); title('Incorrect diff');

figure; 
subplot(1,2,1); 
    boxplot(cumsum(correctSort,2),'colors','b','symbol','b+');
    axis([0.5 8.5 -0.04 1.04]); 
    title('Correct cumulative sum');
subplot(1,2,2); 
    boxplot(cumsum(incorrectSort,2),'colors','k','symbol','ko');
    axis([0.5 8.5 -0.04 1.04]); 
    title('Incorrect cumulative sum');

figure; 
subplot(1,3,1);
    quantileplot(correctSort(:,1:4)); hold on; quantileplot(incorrectSort(:,1:4)); title('Quantile');
subplot(1,3,2); 
    quantileplot(diff(correctSort(:,1:4),1,2)); hold on; quantileplot(diff(incorrectSort(:,1:4),1,2)); title('Quantile diff');
subplot(1,3,3); 
    quantileplot(cumsum(correctSort(:,1:4),2)); hold on; quantileplot(cumsum(incorrectSort(:,1:4),2)); title('Quantile cumulative sum');

% Dendrograms
yyC = pdist(correctSort);
zzC = linkage(yyC,'average');
figure; 
subplot(1,4,2); 
    [hhC,ttC,permC] = dendrogram(zzC,0,'orientation','right'); 
    set(gca,'YDir','reverse','ytick',[]);
subplot(1,4,1); 
    imagesc(correctSort(permC,:)); 
    colormap(hot); 
    caxis([0 1]);
    title('Correct');
    
yyI = pdist(incorrectSort);
zzI = linkage(yyI,'average');
subplot(1,4,4); 
    [hhI,ttI,permI] = dendrogram(zzI,0,'orientation','right'); 
    set(gca,'YDir','reverse','ytick',[]);
subplot(1,4,3); 
    imagesc(incorrectSort(permI,:)); 
    colormap(hot); 
    caxis([0 1]);
    title('Incorrect');

% Diff dendrograms
yyCd = pdist(diff(correctSort,1,2));
zzCd = linkage(yyCd,'average');
figure; 
subplot(1,4,2); 
    [hhCd,ttCd,permCd] = dendrogram(zzCd,0,'orientation','right'); 
    set(gca,'YDir','reverse','ytick',[]);
subplot(1,4,1); 
    imagesc(-1*diff(correctSort(permCd,:),1,2)); 
    colormap(hot); 
    caxis([0 1]);
    title('Correct diff');
    
yyId = pdist(diff(incorrectSort,1,2));
zzId = linkage(yyId,'average');
subplot(1,4,4); 
    [hhId,ttId,permId] = dendrogram(zzId,0,'orientation','right'); 
    set(gca,'YDir','reverse','ytick',[]);
subplot(1,4,3); 
    imagesc(-1*diff(incorrectSort(permId,:),1,2)); 
    colormap(hot); 
    caxis([0 1]);
    title('Incorrect diff');

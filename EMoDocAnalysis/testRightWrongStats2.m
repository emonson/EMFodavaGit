% 1:Difference, 2:Fill, 3:Entropy, 4:Correlation, 5:SrcNeighborCt, 6:WtSrcNeighbors, 7:Correct

% U = csvread('uncert.csv',1,0);
U2 = U(:,[3 4 5 7]);
% Rescale neighbor count column
U2(:,3) = U(:,5)./3;
U2( U2(:,3)>1, 3 ) = 1;

yyU2 = pdist(U2(:,1:4));
zzU2 = linkage(yyU2,'average');

figure; 
subplot(1,2,2);
[hhU2,ttU2,permU2] = dendrogram(zzU2,0,'orientation','right');
set(gca,'YDir','reverse','ytick',[]);

subplot(1,2,1); 
imagesc(U2(permU2,:)); colormap(hot); caxis([0 1]);
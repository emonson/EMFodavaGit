load X20
classes(classes(:,1)==0,:)=[];
classes(:,2)=[];
[yy,ii] = sort(classes);

cd('/Users/emonson/Data/Fodava/DataSets/sn_lda_8');
rr = dlmread('final.beta',' ');
gg = dlmread('final.gamma',' ');

figure; imagesc(gg(ii,:)); colormap(hot);

[gmax, gimax] = max(gg,[],2);
[rmax,rimax] = max(rr,[],1);

figure; plot(classes+0.15*randn(1047,1),gimax+0.15*randn(1047,1),'o','Color',[0 0 0.4]);

[gY,gI] = sortrows([gimax gmax],[1 2]);
figure; imagesc(gg(gI,:)); colormap(hot);

[rY,rI] = sortrows([rimax;rmax]',[1 2]);
figure; imagesc(rr(:,rI)); colormap(hot);
caxis([-10 -4]);

% Display the multiscale decomposition by using sum of scaling functions
% Uses nonnegative scaling functions
figure;
for j = 1:size(Tree,1),
   % Find the most important scaling function by thresholding the diagonal of T
   lMainIdxs = find(abs(diag(Tree{j,1}.T{1}))>0.3);
   % Compute the power of T. Bad way of doing this, ok for small graphs
   T_j = T^(2^j);
   % Sum the important nonnegative scaling functions      
   T_reduced{j} = Tree{j,1}.T{1}(lMainIdxs,lMainIdxs);
   lScalingFcns{j} = T_j(:,Tree{j,1}.ExtIdxs(lMainIdxs));   
   lSumScalingFcns = sum(T_j(:,Tree{j,1}.ExtIdxs(lMainIdxs)),2);
   subplot(3,3,j); 
   scatter(lX_s(1,:),lX_s(2,:),8,lSumScalingFcns,'filled');
   title(sprintf('Level %d (with %d functions)',j,length(lMainIdxs)));hold on;
   %gplot(full(double(abs(Tree{j,1}.T{1}(1:size(vCenters{j},2),1:size(vCenters{j},2)))>1e-6)),vCenters{j}'*[0,-1;1,0]);
end;

% [XI,YI] = meshgrid(-200:1,1:200);
% figure;
% for jj = 1:size(Tree,1),
% 
%     kk = [1 24 1 1 1 1 1 1 1];
%     ZI = griddata(lX_s(1,:),lX_s(2,:),full(lScalingFcns{jj}(:,kk(jj))),XI,YI);
%     % ZI = griddata(lX_s(1,:),lX_s(2,:),full(lScalingFcnsNorm{5}(:,1)),XI,YI);
% 
%     subplot(3,3,jj);
%      plot(lX_s(1,:),lX_s(2,:),'k.');
%      hold on;
% 
%     HI = image(-200:1,1:200,ZI);
%     set(HI,'CDataMapping','scaled');
%     axis image;
%     colormap(hot);
%      set(HI,'AlphaData',0.7*ones(size(ZI)));
%     set(gca, 'Visible', 'off');
% end;
% 
% figure;
% for jj = 1:size(Tree,1),
%     subplot(3,3,jj);
% 
%     kk = [1 24 1 1 1 1 1 1 1];
%     scatter(lX_s(1,:),lX_s(2,:),8,full(lScalingFcns{jj}(:,kk(jj))),'filled');
%     set(gca, 'Visible', 'off');
%     axis image;
%     colormap(hot);
% end;
% 

figure; 
for kk=1:9, 
    subplot(3,3,kk);
    imgtmp = abs(full(T_reduced{kk}).*(1-eye(size(full(T_reduced{kk})))));
    lmaxsim = max(imgtmp(:));
    imagesc(imgtmp./lmaxsim); 
    axis image; 
    colormap(blackNblue); 
    % colorbar; 
    set(gca,'Visible','off');
end;

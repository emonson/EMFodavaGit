% If G needs to be generated, run script_02 first
% script_02;

% Use neato to lay out graph
% [xx, yy, labels] = draw_dot(G.W);

% Plot basis functions on 
jj = 13;
kk = 44;

% Look at the functions at this scale
figure;
imagesc(G.Tree{jj,1}.ExtBasis);
colormap(map2);
balanceColor;
colorbar;

% Plot the graph points colored by the diffusion function
%   and mark point at next scale calculated by various methods
figure; 
% subplot(1,2,1); 
% scatter(xx,yy,50,G.Tree{jj,1}.ExtBasis(:,kk),'filled'); 
% axis image;
% colormap(map3); 
% balanceColor; 
% colorbar;
% 
% subplot(1,2,2);
scatter(G.X(1,:),G.X(2,:),50,G.Tree{jj,1}.ExtBasis(:,kk),'filled'); 
hold on;

lFcn = G.Tree{jj,1}.ExtBasis(:,kk);

% onscaling
posTemp = (lFcn./norm(lFcn,1))'*G.X';
plot(posTemp(1,1), posTemp(1,2), 'ko','LineWidth',2); 
% onscaling (abs)
posTemp = (abs(lFcn)./norm(abs(lFcn),1))'*G.X';
plot(posTemp(1,1), posTemp(1,2), 'k+','LineWidth',2); 
% onscalingsquared
posTemp2 = (lFcn.^2)'*G.X';
plot(posTemp2(1,1), posTemp2(1,2), 'kx','LineWidth',2); 
% tcols (Not working right now...)
% posTemp3 = zeros(1,2);
% posTemp3(1,:) = DWApplyDyadicPower( G.Tree, G.X(1,:)', jj, G.P(:,G.Tree{jj,1}.ExtIdxs)' )';
% posTemp3(2,:) = DWApplyDyadicPower( G.Tree, G.X(2,:)', jj, G.P(:,G.Tree{jj,1}.ExtIdxs)' )';
% plot(posTemp3(1,1), posTemp3(1,2), 'ko','LineWidth',2); 
axis image;
colormap(map3); 
balanceColor; 
colorbar;
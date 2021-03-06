% G needs to be generated first

% Use neato to lay out graph
[xx, yy, labels] = draw_dot(G.W);

% Copy new coordinates into graph structure
G.X = [xx; yy];

% Propagate new coordinate system to all scales
for ii = 1:size(G.Tree,1),
    if ii==0, continue; end;
    G.Tree{ii,1}.X = ((G.Tree{ii,1}.ExtBasis.^2)'*G.X')';
end

% Plot particular basis functions for 
jj = 13;    % Scale
kk = 44;    % Function

% Look at all the functions at this scale as an image
figure;
imagesc(G.Tree{jj,1}.ExtBasis);
colormap(map2);
balanceColor;
colorbar;

% Look at points in graph colored by one particular basis function
figure; 
scatter(xx,yy,50,G.Tree{jj,1}.ExtBasis(:,kk),'filled'); 
axis image;
colormap(map3); 
balanceColor; 
colorbar;

% Step through 
PlotMultiscaleGraphSeries(G,struct('Type','Animation'));
% or in subplots on the same figure
% PlotMultiscaleGraphSeries(G,struct('Type','Array','Scales',[3:15]));
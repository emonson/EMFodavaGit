function DrawSimpleGraph2(G,cOpts)

%
% function DrawSimpleGraph2(G,cOpts)
%
% Plots a graph
%
% IN:
%   G       : structure containing at least the field W and X
%   cOpts   : structure of options:
%               Level      : Level within hierarchy to plot
%               Offset     : Negative offset of level plotted behind points
%               FigHandle  : figure handle
%               MarkerSize : Default: 3.
%               WThres     : do not display edges with similarities smaller
%                            than this threshold. Default: 0.
%               NoiseWiggle: size of noise to add to points. Default: 0.
%               [Edges]    : display or not edges. Default: true.
%
% OUT:
%
%
%
%

% @Copyright 2008
%   Mauro Maggioni
%   Duke University
%
%


if nargin<2,
    cOpts = [];
end;

if ~isfield(cOpts,'Level'),
    cOpts.Level = 1;
end;
if ~isfield(cOpts,'Offset'),
    cOpts.Offset = 3;
end;
if ~isfield(cOpts,'FigHandle'),
    cOpts.FigHandle = figure;
end;
if ~isfield(cOpts,'MarkerSize'),
    cOpts.MarkerSize = 2;
end;
if ~isfield(cOpts,'WThres'),
    cOpts.WThres = 0;
end;
if ~isfield(cOpts,'Edges'),
    cOpts.Edges = true;
end;
if ~isfield(cOpts,'MaxSim'),
    cOpts.MaxSim = 1.0;
end;
if ~isfield(cOpts,'NoiseWiggle'),
    cOpts.NoiseWiggle = 0;
end;

W = G.Tree{cOpts.Level,1}.T{1};
% Coordinates of current scale based on scaling function squared
X = ((G.Tree{cOpts.Level,1}.ExtBasis.^2)'*G.X')';
ExtBasis = G.Tree{cOpts.Level,1}.ExtBasis;

W = W.*(1-eye(size(W,1)));
W = W.*(abs(W) > cOpts.WThresh);

% Find edges
[i,j,s] = find(W);

% Add a tiny bit of random noise
if cOpts.NoiseWiggle>0,
    X = X+cOpts.NoiseWiggle*randn(size(X));
end;

figure(cOpts.FigHandle);
if isfield(cOpts,'Subplot'),
    subplot(cOpts.Subplot);
end

set(gcf, 'Color', 'w');
if ~isfield(cOpts,'Subplot'),
    % If animating, need to completely clear out plot
    clf(cOpts.FigHandle);
end

% Plot background, original fine-scale vertices
plot(G.X(1,:),G.X(2,:),'.','MarkerSize',6,'Color',[0.2 0.4 0.2]);

hold on;
axis image;
set(gca,'Visible','off');

% Plot lines between points which are connected strongly enough
numColors = 64;
lineColorMap = 1-copper(numColors);
fprintf(1,'\tLevel: %d -- Similarity (max/min): (%2.3f, %2.3f)\n', cOpts.Level, max(s), min(s));
for kk = 1:length(i),
    l = line( [X(1,i(kk)),X(1,j(kk))],[X(2,i(kk)),X(2,j(kk))] );            
    set(l,'LineWidth',(5/cOpts.MaxSim)*abs(s(kk)));
    cindex = min([fix((abs(s(kk))/cOpts.MaxSim)*numColors)+1 numColors]);
    set(l, 'Color', lineColorMap(cindex,:));
end;

% Plot red circle markers
plot(X(1,:),X(2,:),'ro','LineWidth',3,'MarkerSize',6,'Color',[181 16 16]./255);

text(0,0,sprintf('Scale %d', cOpts.Level));
hold off;

return;
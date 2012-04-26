figure;
hold on;

GWT = gW;
leafNodes = find(leafnodes( GWT.cp ));
col = 1.0*ones(1,3);
minSc = 2;
maxSc = 5;

for j = minSc:maxSc,
    linewidth = 1.5*(7-j);
    nets = sort([find(GWT.Scales == j) leafNodes(GWT.Scales(leafNodes)<j)], 'ascend');
    disp(nets);
    lens  = cumsum(GWT.Sizes(nets));
    for i = 1:length(lens)-1
        ln = line([lens(i) lens(i)], [0.5 8.5]); set(ln,'color',col.*((j-minSc)/maxSc),'LineWidth',linewidth);
    end
end

scatter(1:1047, labels(gW.PointsInNet{end},2)+(0.4*(rand(1047,1)-0.5)), 30, labels(gW.PointsInNet{end},2), 'LineWidth',1.5);
axis([1 1047 0 9]);
colormap(brewerDark1);
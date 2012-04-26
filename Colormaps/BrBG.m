function h = BrBG(m)
% BrBG colormap
%
% From ColorBrewer2
% http://colorbrewer2.org/index.php?type=diverging&scheme=BrBG&n=7

if nargin < 1, m = size(get(gcf,'colormap'),1); end

yy = [140, 81, 10; 
      216, 179, 101; 
      246, 232, 195; 
      245, 245, 245; 
      199, 234, 229; 
      90, 180, 172; 
      1, 102, 94];

x0 = 1:-2/(size(yy,1)-1):-1;

x1 = 1:-(2/(m-1)):-1;

h = interp1(x0, yy, x1, 'linear')/255;

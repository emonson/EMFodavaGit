function h = RdGy(m)
% RdGy colormap
%
% From ColorBrewer2
% http://colorbrewer2.org/index.php?type=diverging&scheme=RdGy&n=7

if nargin < 1, m = size(get(gcf,'colormap'),1); end

yy = [178, 24, 43; 
      239, 138, 98; 
      253, 219, 199; 
      255, 255, 255; 
      224, 224, 224; 
      153, 153, 153; 
      77, 77, 77]; 

x0 = 1:-2/(size(yy,1)-1):-1;

x1 = 1:-(2/(m-1)):-1;

h = interp1(x0, yy, x1, 'linear')/255;

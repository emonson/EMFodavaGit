function h = RdBu(m)
% RdBu colormap
%
% From ColorBrewer2
% http://colorbrewer2.org/index.php?type=diverging&scheme=RdBu&n=7

if nargin < 1, m = size(get(gcf,'colormap'),1); end

yy = [178 24 43; ...
      239 138 98; ...
      253 219 199; ...
      247 247 247; ...
      209 229 240; ...
      103 169 207; ...
      33 102 172];

x0 = 1:-2/(size(yy,1)-1):-1;

x1 = 1:-(2/(m-1)):-1;

h = interp1(x0, yy, x1, 'linear')/255;

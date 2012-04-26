function c = brewerSet1(m)
% Set1 discrete colormap from colorbrewer.org
%
% Not a generic colormap since it is limited to 9 colors

if nargin < 1, m = 9; end

if (m > 9), m = 9; end;

tmp = [228	26	28;
        55	126	184;
        77	175	74;
        152	78	163;
        255	127	0;
        255	255	51;
        166	86	40;
        247	129	191;
        153	153	153]./256;

 c = tmp(1:m,:);

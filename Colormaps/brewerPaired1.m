function c = brewerPaired1(m)
% Set1 discrete colormap from colorbrewer.org
%
% Not a generic colormap since it is limited to 12 colors

if nargin < 1, m = 12; end

if (m > 12), m = 12; end;

tmp = [166	206	227;
        31	120	180;
        178	223	138;
        51	160	44;
        251	154	153;
        227	26	28;
        253	191	111;
        255	127	0;
        202	178	214;
        106	61	154;
        255	255	153;
        177	89	40;]./256;

 c = tmp(1:m,:);

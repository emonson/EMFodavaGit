function c = brewerDark1(m)
% Set1 discrete colormap from colorbrewer.org
%
% Not a generic colormap since it is limited to 8 colors

if nargin < 1, m = 8; end

if (m > 8), m = 8; end;

tmp = [27	158	119;
        217	95	2;
        117	112	179;
        231	41	138;
        102	166	30;
        230	171	2;
        166	118	29;
        102	102	102;]./256;

 c = tmp(1:m,:);

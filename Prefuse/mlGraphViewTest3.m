% javaaddpath /Users/emonson/Programming/Eclipse3.4workspace/FodavaLabelTest/bin;
clear all;
javarmpath /Users/emonson/Programming/Eclipse3.4workspace/FodavaLabelTest/bin;
javaaddpath /Users/emonson/Programming/Eclipse3.4workspace/FodavaLabelTest/bin;
% cd('/Users/emonson/Programming/Python/VTK');
% load('G_010809.mat');
% script_02_larger;

baseName = 'X20_042709c';

cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
fprintf(1,'Loading data set\n');
    load([baseName '.mat']);
    pts = dlmread([baseName '_pts.csv'],' ');

import edu.duke.vis.fodavatest.GraphViewTest_fromMatlab1;
import edu.duke.vis.fodavatest.GraphPCombo;
import javax.swing.JFrame;

scale = 4;
% [ii,jj,vv] = find(G.W);
extSum = sum(G.Tree{scale,1}.ExtBasis,1);
extLogical = extSum > 0;
tmp = G.Tree{scale,1}.T{1}(extLogical,extLogical);

% Don't prefilter weights
% [ii,jj,vv] = find(tmp);

% Pre-filter weights
II = find(tmp > 0);
[ii,jj] = ind2sub(size(tmp),II);
vv = tmp(II);

% [ii,jj,vv]
% Construction of P matrix assumes zero indexed values...
gP = GraphPCombo(ii-1, jj-1, vv);


view = GraphViewTest_fromMatlab1(gP.getG(), gP.getP());

frame = JFrame('p r e f u s e  |  g r a p h v i e w');
% frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
frame.setContentPane(view);
frame.pack();
frame.setVisible(true);

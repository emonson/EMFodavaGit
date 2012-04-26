clear all;
javarmpath /Users/emonson/Programming/Eclipse3.4workspace/FodavaLabelTest/bin;
javaaddpath /Users/emonson/Programming/Eclipse3.4workspace/FodavaLabelTest/bin;
cd('/Users/emonson/Programming/Matlab/EMonson/Fodava/DocumentAnalysis/Analysis');
load('X20_042709b.mat','G');

import edu.duke.vis.fodavatest.GraphViewTest_fromMatlab1;
import edu.duke.vis.fodavatest.GraphPCombo;
import javax.swing.JFrame;

% T = abs(G.Tree{7,1}.T{1});
% kk = 2;
% b = 1;
% find subset of T which is connected
% while (b > 0),
%     b=connected_graph(T(1:kk,1:kk)>0,0); 
%     fprintf(1,'%d: %d\n', kk, b); 
%     kk = kk + 1;
% end
% kk = kk - 2;
% Tsub = T(1:kk,1:kk);
% [ii,jj,vv] = find(Tsub);

Tsub = G.T./10;

% Pre-filter weights
II = find(Tsub > 0);
[ii,jj] = ind2sub(size(Tsub),II);
vv = Tsub(II);

% [ii,jj,vv]
% Construction of P matrix assumes zero indexed values...
gP = GraphPCombo(ii-1, jj-1, vv);


view = GraphViewTest_fromMatlab1(gP.getG(), gP.getP());

frame = JFrame('p r e f u s e  |  g r a p h v i e w');
% frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
frame.setContentPane(view);
frame.pack();
frame.setVisible(true);

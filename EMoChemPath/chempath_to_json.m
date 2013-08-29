% Script to transfer Miles' chem simulation space and path data to JSON
% for visualization with the interactive D3 ellipses & paths

load('d_info.mat');
savejson('',d_info,'ArrayToStruct',1,'ArrayIndent',0,'FileName','d_info.json','ForceRootName',0);

clear all;

load('netpoints.mat');
savejson('',netpoints,'ArrayToStruct',1,'ArrayIndent',0,'FileName','netpoints.json','ForceRootName',0);

clear all;

load('traj1.mat');
savejson('',sim_opts,'ArrayToStruct',1,'ArrayIndent',0,'FileName','sim_opts.json','ForceRootName',0);

trajectory = struct();
trajectory.path = path;
trajectory.path_index = path_index;
trajectory.t = t;
% trajectory.v_norm = v_norm;
savejson('',trajectory,'ArrayToStruct',1,'ArrayIndent',0,'FileName','trajectory.json','ForceRootName',0);

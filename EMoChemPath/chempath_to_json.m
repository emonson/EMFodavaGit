% Script to transfer Miles' chem simulation space and path data to JSON
% for visualization with the interactive D3 ellipses & paths

load('d_info.mat');
% NOTE: lmks is a very large sparse matrix that we're not using in the vis right
%   now...
for ii=1:length(d_info),
    d_info{ii} = rmfield(d_info{ii}, 'lmks');
end
savejson('',d_info,'ArrayToStruct',1,'ArrayIndent',0,'FileName','d_info.json','ForceRootName',0);
fprintf('d_info done\n');

clear all;

load('net.mat');
savejson('',net,'ArrayToStruct',1,'ArrayIndent',0,'FileName','netpoints.json','ForceRootName',0);

clear all;

load('traj1.mat');
savejson('',sim_opts,'ArrayToStruct',1,'ArrayIndent',0,'FileName','sim_opts.json','ForceRootName',0);

trajectory = struct();
trajectory.path = path;
trajectory.path_index = path_index;
trajectory.t = t;
% trajectory.v_norm = v_norm;
savejson('',trajectory,'ArrayToStruct',1,'ArrayIndent',0,'FileName','trajectory.json','ForceRootName',0);

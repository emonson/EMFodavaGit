%% Method 1 -- grabbing all wavelet coefficients for a given data index

pt_idx = 2000;
leaf_node = gW.IniLabels(pt_idx);
imap(gW.LeafNodes) = 1:length(gW.LeafNodes);

wav_coeffs_mat = [Data.CelWavCoeffs{imap(leaf_node),:}];
wav_coeffs_vec = wav_coeffs_mat((gW.PointsInNet{leaf_node} == pt_idx),:);

%% Method 2 -- also builds a list (cell array) of the coeffs at each scale

pt_idx = 2000;
leaf_node = gW.IniLabels(pt_idx);
pt_row = find(gW.PointsInNet{leaf_node} == pt_idx);
imap(gW.LeafNodes) = 1:length(gW.LeafNodes);
leaf_idx = imap(leaf_node);

scales = size(Data.CelWavCoeffs,2);
wav_coeffs = cell(1,scales);

for ii = 1:scales,
    if(~isempty(Data.CelWavCoeffs{leaf_idx,ii}))
        wav_coeffs{ii} = Data.CelWavCoeffs{leaf_idx,ii}(pt_row,:);
    else
        wav_coeffs{ii} = [];
    end
end

wav_coeffs_vec = [wav_coeffs{:}];

%% Constructing all the wavelet coeffs for a given node

node_idx = 32;
offspring = get_offspring(gW.cp, node_idx);
offspring = [node_idx offspring];
relevantLeafNodes = offspring(logical(gW.isaleaf(offspring)));
imap(gW.LeafNodes) = 1:length(gW.LeafNodes);
wav_coeffs_img = cat(1, Data.CelWavCoeffs{imap(relevantLeafNodes), gW.Scales(n)});

%% Build cell array of already constructed wavelet coeffs for each node

num_nodes = length(gW.cp);
imap(gW.LeafNodes) = 1:length(gW.LeafNodes);
wav_coeff_imgs = cell(1,num_nodes);

for node_idx = 1:num_nodes,
    offspring = [node_idx get_offspring(gW.cp, node_idx)];
    % offspring = [node_idx offspring];
    relevantLeafNodes = offspring(logical(gW.isaleaf(offspring)));
    wav_coeff_imgs{node_idx} = cat(1, Data.CelWavCoeffs{imap(relevantLeafNodes), gW.Scales(node_idx)});
end

%% Rotating MNIST digits for GUI

N = 1000;    % orig = 5000
X0 = Generate_MNIST([N], struct('Sampling', 'RandN', 'QueryDigits', [1 2], 'ReturnForm', 'vector'));
% In the vis GUI, the numbers end up rotated, so here do the transformation
% before the data analysis to get them to end up the correct direction
yy = reshape(X0,[],28,28);
yy1 = permute(yy,[1 3 2]);
yy2 = flipdim(yy1,3);
X0 = reshape(yy2,[],784);
clear('yy','yy1','yy2');
imR = 28;

% Set the number of dimensions for each plane (used as "intrinsic
% dimensionatiy" of dataset in geometric_wavelets_transformation
% k = 3;

WRITE_OUTPUT = true;
data_dir = '/Users/emonson/Data/Fodava/EMoGWDataSets';
out_file = 'mnist12_1k_20100624.mat';

%% Saving quantities for GUI (fixed dim original version)

if (WRITE_OUTPUT)
    cd(data_dir);
    save(out_file,'Data','gW','V','J','nAllNets','leafNodes','X0','X','WavCoeffs','WavCoeffMags','imR','imC');
end

%% Saving quantities for GUI use

if (WRITE_OUTPUT)
    cd(data_dir);
    
    % "poofing" out variables for easier loading in Python code 
    % and for file size since don't need nearly all of this stuff...
    % I know it makes it less flexible later...

    AmbientDimension = GWTopts.AmbientDimension;
    X = gW.X;
    % cm
    % imR
    % imC
    
    % Redundant for now...
    CelWavCoeffs = Data.CelWavCoeffs;
    
    num_nodes = length(gW.cp);
    LeafNodesImap(gW.LeafNodes) = 1:length(gW.LeafNodes);
    NodeWavCoeffs = cell(1,num_nodes);

    for node_idx = 1:num_nodes,
        offspring = [node_idx get_offspring(gW.cp, node_idx)];
        relevantLeafNodes = offspring(logical(gW.isaleaf(offspring)));
        NodeWavCoeffs{node_idx} = cat(1, Data.CelWavCoeffs{LeafNodesImap(relevantLeafNodes), gW.Scales(node_idx)});
    end
    
    CelScalCoeffs = Data.CelScalCoeffs;
    NodeScalCoeffs = cell(1,num_nodes);

    for node_idx = 1:num_nodes,
        offspring = [node_idx get_offspring(gW.cp, node_idx)];
        relevantLeafNodes = offspring(logical(gW.isaleaf(offspring)));
        NodeScalCoeffs{node_idx} = cat(1, Data.CelWavCoeffs{LeafNodesImap(relevantLeafNodes), gW.Scales(node_idx)});
    end
    
    % Should calculate Projections rather than storing -- it's big...
    
    % node_idx = leafNodes(leaf_node_idx);
    % data_idxs = find(gW.IniLabels == node_idx); % same as PointsInNet{net}
    % nPts = length(data_idxs);
    % j_max = gW.Scales(node_idx);
    % gWCentersnet = repmat(gW.Centers{node_idx},nPts,1);
    % Data.Projections(data_idxs,:,j_max) = Data.CelScalCoeffs{i,j_max}*gW.ScalFuns{node_idx}' + gWCentersnet;
    % X_approx = Data.Projections(:,:,scale);
    % X_img = X_approx*V(:,1:GWTopts.AmbientDimension)'+repmat(cm, size(X_approx,1),1);

    % Projections = Data.Projections;
    
    % May be able to get away with only saving
    % V(:,1:GWTopts.AmbientDimension)
    V = Data.V(:,1:GWTopts.AmbientDimension);
    
    cp = gW.cp;
    IniLabels = gW.IniLabels;
    PointsInNet = gW.PointsInNet;
    NumberInNet = gW.Sizes;
    ScalFuns = gW.ScalFuns;
    WavBases = gW.WavBases;
    Centers = gW.Centers;
    Scales = gW.Scales;
    IsALeaf = gW.isaleaf;
    LeafNodes = gW.LeafNodes;

    save(out_file,...
        'cm','imR','imC','AmbientDimension','X','CelWavCoeffs',...
        'LeafNodesImap','NodeWavCoeffs','CelScalCoeffs',...
        'NodeScalCoeffs','Projections','V','cp','IniLabels',...
        'PointsInNet','NumberInNet','ScalFuns','WavBases',...
        'Centers','Scales','IsALeaf','LeafNodes','GWTopts');

end

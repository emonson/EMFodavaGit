function G = script_02_test( )

% Set parameters for building a multiscale graph
GraphType={'Tree','Complete','Tree','Complete'};
% Set options for each of the levels of graphs
% Top level
GraphOpts{1}=struct('undirected',1,'K',2,'Levels',3);
% Two levels down
GraphOpts{2}=struct('undirected',1,'N',3);
GraphOpts{3}=struct('undirected',1,'K',2,'Levels',1);
GraphOpts{4}=struct('undirected',1,'N',3);

% Set options for the graph hierarchy
Opts.NumberOfLevels=4;
Opts.Weights=[1,2,4,8];
%Opts.Weights=[1,1.2,1.4,1.6];
Opts.GraphOpts=GraphOpts;Opts.Embed = true;

% Create the multiscale graph
G = CreateMultiscaleGraph( GraphType, Opts );

% Add noise to the graph
Go = G;
G = AddNoiseToGraph( G, struct('NoiseSize',0.02) );
%G = AddNoiseToGraph( G, struct('NoiseSize',0.1,'NoiseType','AddEdges','EdgesN',5) );
%G = AddNoiseToGraph( G, struct('NoiseSize',0.1,'NoiseType','RemoveEdges','EdgesN',10) );

figure;imagesc(G.W-Go.W);colorbar; title('Difference between the two similarity matrices');

% Compute operators on the graph
G = ConstructGraphOperators ( G );

% Compute the eigenvalues and eigenfunctions
[G.EigenVecs,G.EigenVals] = eigs(G.T,min([100,size(G.W,1)]),'LM',struct('verbose',0));
G.EigenVals = diag(G.EigenVals);

% Display the spectrum
figure;plot(G.EigenVals);grid on;axis tight;title('EigenValues of T');

% Compute the multiscale analysis
WaveletOpts = struct('Symm',true,'Wavelets',false); % ,'GS','gsqr_qr');
G.Tree = DWPTree (G.T, 40, 1e-12, WaveletOpts);


return;
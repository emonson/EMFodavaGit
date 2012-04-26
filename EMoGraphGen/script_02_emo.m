% % Set parameters for building a multiscale graph
% GraphType={'Tree','Complete','Tree','Complete'};
% % Set options for each of the levels of graphs
% % Top level
% GraphOpts{1}=struct('undirected',1,'K',2,'Levels',3);
% % Two levels down
% GraphOpts{2}=struct('undirected',1,'N',3);
% GraphOpts{3}=struct('undirected',1,'K',2,'Levels',1);
% GraphOpts{4}=struct('undirected',1,'N',3);

% Set parameters for building a multiscale graph
GraphType={'Complete','Tree','Complete','Tree'};
% Set options for each of the levels of graphs
% Top level
GraphOpts{1}=struct('undirected',1,'N',3);
% Two levels down
GraphOpts{2}=struct('undirected',1,'K',2,'Levels',3);
GraphOpts{3}=struct('undirected',1,'N',3);
GraphOpts{4}=struct('undirected',1,'K',2,'Levels',2);

% Set options for the graph hierarchy
Opts.NumberOfLevels=4;
Opts.Weights=[1,2,4,8];
%Opts.Weights=[1,1.2,1.4,1.6];
Opts.GraphOpts=GraphOpts;Opts.Embed = true;

% Create the multiscale graph
G = CreateMultiscaleGraph( GraphType, Opts );

% Add noise to the graph
Go = G;
%G = AddNoiseToGraph( G, struct('NoiseSize',1) );
%G = AddNoiseToGraph( G, struct('NoiseSize',0.1,'NoiseType','AddEdges','EdgesN',5) );
%G = AddNoiseToGraph( G, struct('NoiseSize',0.1,'NoiseType','RemoveEdges','EdgesN',10) );

figure;imagesc(G.W-Go.W);colorbar; title('Difference between the two similarity matrices');

% Simple layout
DrawSimpleGraph(G,struct('NoiseWiggle',0.05));

% Graphviz layout
%BGobj=biograph(G.W);BGobj.view;

% Compute operators on the graph
G = ConstructGraphOperators ( G );

% Compute the eigenvalues and eigenfunctions
[G.EigenVecs,G.EigenVals] = eigs(G.T,min([100,size(G.W,1)]),'LM',struct('verbose',0));
G.EigenVals = diag(G.EigenVals);

% Display the spectrum
figure;plot(G.EigenVals);grid on;axis tight;title('EigenValues of T');

% Display some eigenvectors
% for k = min([20,length(G.EigenVals)]):-1:1,
%     DrawSimpleGraph(G,struct('Fcn',G.EigenVecs(:,k)));title(sprintf('Eigenvector %d',k));
% end;

% Compute the multiscale analysis
WaveletOpts = struct('Symm',true,'Wavelets',false); %,'GS','gsqr_qr');
G.Tree = DWPTree (G.T, 40, 1e-12, WaveletOpts);

% Display some of the scaling functions, pointwise squared
% figure;
% for j = 3:size(G.Tree,1)
%     for k = 1:size(G.Tree{j,1}.ExtBasis,2),
%         scatter(G.X(1,:),G.X(2,:),50,G.Tree{j,1}.ExtBasis(:,k),'filled');colorbar;
%         % DrawSimpleGraph(G,struct('Fcn',G.Tree{j,1}.ExtBasis(:,k)));
%         title(sprintf('Scaling function (%d,%d)',j,k));pause;
%     end;
% end;

% Display the wavelet tree
% DrawSimpleMultiscale(G);

%G = DrawSimpleMultiscale( G,struct('Type','ONScalingSquared','GraphOpts',struct('MarkerSize',0,'Edges',false,'NoiseWiggle',0),'Scales',[0:17],'Representation',struct('Type','Animation')) );

% This is even better! No need to do the funny squaring above
%G = DrawSimpleMultiscale( G,struct('Type','onscalingsquared','GraphOpts',struct('MarkerSize',0,'Edges',false,'NoiseWiggle',0),'Scales',[0:19],'Representation',struct('Type','Animation')) );
G = DrawSimpleMultiscale( G,struct('Type','onscalingsquared','GraphOpts',struct('MarkerSize',0,'Edges',false,'NoiseWiggle',0),'Scales',[0:19],'Representation',struct('Type','Animation')) );

    
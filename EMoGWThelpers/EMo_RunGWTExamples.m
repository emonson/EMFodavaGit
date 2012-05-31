% clear all
% close all
% clc

%% Go parallel
% if matlabpool('size')==0,
%     matlabpool
% end;

%% Pick a data set
pExampleNames  = {'MNIST_Digits','YaleB_Faces','croppedYaleB_Faces',...
                  'Frey_faces','Olivetti_faces','NaturalImagePatches',...
                  'Medical12images','Medical12Sift','CorelImages','CorelSift',...
                  'ScienceNews', 'ScienceNewsTFIDF', 'ScienceNewsCounts',...
                  '20NewsSubset1', '20NewsSubset2tf', '20NewsSubset2tfidf', '20NewsSubset3', '20NewsSubset4',...
                  'IntegralOperator','MeyerStaircase', ...
                  'SwissRoll','S-Manifold','Oscillating2DWave',...
                  'D-Ball','D-Sphere', 'D-Cube','D-FlatTorus','Cosine','Signal1D_1','D-Gaussian'};

fprintf('\n Examples:\n');
for k = 1:length(pExampleNames),
    fprintf('\n [%d] %s',k,pExampleNames{k});
end;
fprintf('\n\n  ');

pExampleIdx = input('Pick an example to run: \n');

%% choose a GWT version
GWTversions  = {'Vanilla GWT','Orthogonal GWT','Pruning GWT'};
methodLabels = [0 1 2];
fprintf('\n Geometric Wavelets version:\n');
for k = 1:length(GWTversions),
    fprintf('\n [%d] %s',methodLabels(k),GWTversions{k});
end;
fprintf('\n\n  ');

pGWTversion = input('Pick a version of the GWT to run: \n');

%% generate data and choose parameters for GWT
[X, GWTopts, imgOpts] = GenerateData_and_SetParameters(pExampleNames{pExampleIdx});

%% construct geometric wavelets
GWTopts.GWTversion = pGWTversion;
GWT = GMRA(X, GWTopts);

%% compute all wavelet coefficients 
[GWT, Data] = GWT_trainingData(GWT, X);

%% display results
GWT_displayResults(GWT,Data,imgOpts);

%% Save data

file_name = input('Enter a file name to save data (without .mat on the end)\n', 's');

if ~isempty(file_name)
    S = GWT_saveGUIdata(pExampleNames{pExampleIdx}, GWT, Data, imgOpts, [file_name '.mat']);
end;

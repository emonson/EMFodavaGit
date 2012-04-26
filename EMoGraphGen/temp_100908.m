function temp_100908(G, cOpts)
% Plot multiple scale graphs with edge colors and cutoff

FF = figure; 

depth = size(G.Tree,1);

if ~isfield(cOpts,'Type'),
    cOpts.Type = 'Animation';
end;
if ~isfield(cOpts,'Scales'),
    cOpts.Scales = 1:size(G.Tree,1);
end;


switch lower(cOpts.Type),
    case {'animation'}
        for jj = cOpts.Scales,
            DrawSimpleGraph2(G,struct('Level',jj,'MaxSim',0.12,'WThresh',0.005,'FigHandle',FF,'Offset',0)); 
            hold off;
            if jj < max(cOpts.Scales),
                pause;
            end;
        end;
    case {'array'}
        
        % TODO: precalc good subplot ranges
        lx = 4;
        ly = 5;
        ii = 1;

        for jj = cOpts.Scales, 
            subplot(ly,lx,ii); 
            DrawSimpleGraph2(G,struct('Level',jj,'MaxSim',0.12,'WThresh',0.005,'FigHandle',FF)); 
            ii = ii + 1;
        end;
    otherwise
        fprintf('DrawSimpleMultiscale: representation %s unknown.\n',cOpts.Representation);        
end;


return;
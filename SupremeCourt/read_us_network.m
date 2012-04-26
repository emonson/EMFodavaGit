cd('/Volumes/SciVis_LargeData/ArtMarkets/judicial/');

out_dir = '/Users/emonson/Programming/ArtMarkets/SupremeCourt/';

h = figure(1000);
hold on;

case_years = dlmread('case_years.txt','\t');
edges = dlmread('allcites.txt',' ');

caseid = case_years(:,1);
year = case_years(:,2);
edge_year = year(edges(:,1));

tmin = 1880;
tinc = 10;
tmax = 2000;
count = 0;

for t = tmin:tinc:tmax

    fprintf(1, 'Loading year = %d : ', t);

    edges_t = edges(edge_year <= t, :);
    N = max(max(edges_t));
    W_t = sparse(edges_t(:,1), edges_t(:,2), 1.0, N, N);

    % Do simplistic (mirror) for symmetry (undirected graph)
    W_t = W_t + W_t';

    % Put in self-loops
    % W_t = W_t + sparse(1:max_id, 1:max_id, 1.0);

    % Add diagonal elements into the edge matrix
    % equal to the node degree
    D = sum(W_t, 2);
    W = W_t;
    % degree(degree == 0) = 1;
    W_t = -1.*W_t + sparse(1:N, 1:N, D);

    % Normalize rows
    % degree = sum(W_t, 2);
    % [ii,jj,vv] = find(W_t);
    % vv = vv./degree(ii);
    % W_t = sparse(ii,jj,vv);
    % clear('ii','jj','vv');

    Dinv = D.^(-1);
    Dinv(isinf(Dinv)) = 0;
    Dinvsqrt = Dinv.^(1/2);

    Dinv = spdiags(Dinv, 0, N, N);
    Dinvsqrt = spdiags(Dinvsqrt, 0, N, N);

    L_t = Dinvsqrt * W_t * Dinvsqrt; 
    P = Dinv*W;
    P_t = Dinv * W_t;

    % [EVec, EVal] = eigs(L_t, 200);
    % [EVec, EVal] = eig(full(L_t));
    % [EVec, EVal] = GetEigs(L_t, 100, P_t, struct('TakeDiagEigenVals',1));

    % Find the number of connected components in the graph
    [numComp, compIdx] = graphconncomp(L_t);

    % Pull out the largest connected component
    comp_count = full(sparse( compIdx, 1, 1 ));
    [Y,I] = max(comp_count);
    fprintf(1, 'count = %d\n', Y);
    Lar_t = L_t( compIdx == I, compIdx == I );
    Par_t = P_t( compIdx == I, compIdx == I );
    Par = P( compIdx == I, compIdx == I );
    
    [rr,cc,vv] = find(Par);
    out_file = [out_dir 'us_P_' num2str(t) '.dl'];
    fid = fopen(out_file, 'w');
    fprintf(fid, 'dl\n');
    fprintf(fid, 'format=edgelist1\n');
    fprintf(fid, 'n=%d\n', length(rr));
    fprintf(fid, 'data:\n');
    fclose(fid);
    dlmwrite(out_file, [rr cc vv], '-append', 'delimiter', ' ');

    fprintf(1, '\tTaking eigenvalues\n');
    [EVec2, EVal2] = GetEigs(Lar_t, 100, Par_t, struct('TakeDiagEigenVals',1));
    
    plot(EVal2 + count, 'Color', [(t - tmin)/(tmax - tmin) 0 0]);
    drawnow;
    count = count + 0.1;
end


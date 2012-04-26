NN = 5;
qq = 0.001;

GT = G.T.*(G.T>qq);

% Check if any rows/columns of YY are too sparse
fprintf(1,'Adjusting neighbors\n');
for ii = find(sum(GT>0,1) < NN),
    % Add elements from XX back into YY to reach required NN count
    [sC,sI] = sort(G.T(:,ii),'descend');
    GT(sI(1:NN),ii) = sC(1:NN);
    GT(ii,sI(1:NN)) = sC(1:NN);
end;

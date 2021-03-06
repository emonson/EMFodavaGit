function hmap = struct2hashmap(S)

% http://stackoverflow.com/questions/436852/storing-matlab-structs-in-java-
% objects

% >> M = java.util.HashMap;
% >> M.put(1,'a');
% >> M.put(2,33);
% >> s = struct('a',37,'b',4,'c','bingo')
% 
% s = 
%     a: 37
%     b: 4
%     c: 'bingo'
% 
% >> M.put(3,struct2hashmap(s));
% >> M
% 
% M =
% {3.0={a=37.0, c=bingo, b=4.0}, 1.0=a, 2.0=33.0}

    if ((~isstruct(S)) || (numel(S) ~= 1)),
        error('struct2hashmap:invalid','%s',...
              'struct2hashmap only accepts single structures');
    end

    hmap = java.util.HashMap;
    for fn = fieldnames(S)'
        % fn iterates through the field names of S
        % fn is a 1x1 cell array
        field = fn{1};
        value = S.(field);
        if isstruct(value),
            innerMap = struct2hashmap(value);
            hmap.put(field, innerMap);
        else
            hmap.put(field,value);
        end
    end
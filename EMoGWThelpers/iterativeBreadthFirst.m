function iterativeBreadthFirst(tree_parent_idxs, test_array, test_function)
% http://en.wikipedia.org/wiki/Tree_traversal

% Start at root of the tree (cp(root_idx) == 0)
root_idx = find(tree_parent_idxs == 0);

if (length(root_idx) > 1)
    fprintf( 'cp contains too many root nodes (cp == 0)!! \n' );
    return;
else
    % The java deque works with First = most recent, Last = oldest
    % so since it can be accessed with removeFirst / removeLast
    % it can be used either as a LIFO stack or FIFO queue
    % Here I'm trying it as a deque/queue
    nodes = java.util.ArrayDeque();
    nodes.addFirst(root_idx);

    while (~nodes.isEmpty())
        current_node = nodes.removeLast();
        
        % Do the actual work on the tree
        fprintf( 'current node: %d\n', current_node );
        
        % Only addFirst children on to the stack if this node qualifies
        if test_function(test_array(current_node))
            % Find childrent of current node
            child_idxs = find(tree_parent_idxs == current_node);

            for idx = child_idxs
                nodes.addFirst(idx);
            end
        end
    end
end
    
end
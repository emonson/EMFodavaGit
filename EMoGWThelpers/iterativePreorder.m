function iterativePreorder(tree_parent_idxs, test_array, test_function)
% http://en.wikipedia.org/wiki/Tree_traversal

% Start at root of the tree (cp(root_idx) == 0)
root_idx = find(tree_parent_idxs == 0);

if (length(root_idx) > 1)
    fprintf( 'cp contains too many root nodes (cp == 0)!! \n' );
    return;
else
    nodes = java.util.Stack();
    nodes.push(root_idx);

    while (~nodes.isEmpty())
        current_node = nodes.pop();
        
        % Do the actual work on the tree
        fprintf( 'current node: %d\n', current_node );
        
        % Only push children on to the stack if this node qualifies
        if test_function(test_array(current_node))
            % Find childrent of current node
            child_idxs = find(tree_parent_idxs == current_node);

            for idx = child_idxs
                nodes.push(idx);
            end
        end
    end
end
    
end
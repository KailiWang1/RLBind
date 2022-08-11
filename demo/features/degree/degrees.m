% INPUTS: adjacency matrix
% OUTPUTS: degree, indegree and outdegree sequences

function [deg,indeg,outdeg]=degrees(adj)

indeg = sum(adj);
outdeg = sum(adj');

if isdirected(adj)
  deg = indeg + outdeg; % total degree

else   % undirected graph: indeg=outdeg
  deg = indeg + diag(adj)';  % add self-loops twice, if any

end
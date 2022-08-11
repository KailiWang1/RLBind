% Using the matrix transpose function
% INPUTS: adjacency matrix
% OUTPUTS: boolean variable

function S=isdirected(adj)

S = true;
if adj==transpose(adj); S = false; end

% one-liner alternative: S=not(issymmetric(adj));
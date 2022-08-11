function [ ] = closeness( input)
%CLOSE_CENTRALITY Summary of this function goes here
%   Detailed explanation goes here

data=load('Outputs/1uud_contact.dat');
G=sparse(data);
[dist] = graphallshortestpaths(G);

cc=zeros(input,1);
for i=1:input
    cc(i) = sum(dist(i,:));% the average distance from the given node to other nodes
end
                 cc=1./cc;
                 cc=cc*28;
   
[x,y]=sort(cc,'ascend');

dlmwrite('Outputs/1uud_closeness.txt', cc, 'precision', '%i', 'delimiter', '\t')

end


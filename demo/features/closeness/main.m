% 
% INPUTS: 
%   PDB file
%   Distacne cutoff 
%
% OUTPUTS:
%   mapping.txt 
%   contact.dat 
%   closeness.txt 

tic
clear

% Input PDB file and distance cutoff
gnetwork('Inputs/1uud.pdb', 8) 

% Here, we identify the number of amino acid in the PDB file
% Open input PDB file
fid = fopen('Inputs_19/1uud.pdb', 'rt');
  
  PDB = textscan(fid, '%s %d %s %s %s %d %f %f %f %f %f %s', 'CollectOutput', true);
  % a vector that stores the NO. of amino acid for each atom
  NO_aminoacid = PDB{4};
  % the number of atoms
  number_of_atom = length(PDB{1});
  
  %count the number of amino acid in PDB file 
  number_of_aminoacid = 1;
  Rev_NO_aminoacid = ones(number_of_atom,1);
  for  i = 2 : number_of_atom
      if abs(NO_aminoacid(i)-NO_aminoacid(i-1)) > 0
          number_of_aminoacid = number_of_aminoacid + 1;
          Rev_NO_aminoacid(i : number_of_atom) = number_of_aminoacid;
      end
  end
  
closeness(number_of_aminoacid)

clear

toc

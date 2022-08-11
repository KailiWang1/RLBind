%
% INPUTS: 
%   PDB file
%   Distacne cutoff
%
% OUTPUTS:
%   mapping.txt
%   contact.dat

  % The input PDB file
  fid = fopen(input, 'rt');
  % The input distance cutoff (between any heavy atoms)
  connect_cutoff = cutoff;
  
  % Sacn PDB file
  PDB = textscan(fid, '%s %d %s %s %s %d %f %f %f %f %f %s', 'CollectOutput', true);
  % The number of atoms
  number_of_atom = length(PDB{1});
  % a vector that stores the NO. for each atom
  NO_atom = PDB{2};
  % a vector that stores the NO. of amino acid for each atom
  NO_aminoacid = PDB{4};
  
  % Count the number of amino acid and get a new(Revised) vector 
  % to denote the NO. of amino acid for each atom
  number_of_aminoacid = 1;
  Rev_NO_aminoacid = ones(number_of_atom,1);
  for  i = 2 : number_of_atom
      if abs(NO_aminoacid(i)-NO_aminoacid(i-1)) > 0
          number_of_aminoacid = number_of_aminoacid + 1;
          Rev_NO_aminoacid(i : number_of_atom) = number_of_aminoacid;
      end
  end
  % get (x,y,z) coordinates of each atom
  position = PDB{5};
  position = position(:,1:3);
  net = zeros(number_of_aminoacid, number_of_aminoacid);
  for i = 1 : number_of_atom
      for j = 1 : number_of_atom
          % We do not consider the themsleves or neighbor residues 
          if abs(Rev_NO_aminoacid(i)-Rev_NO_aminoacid(j)) <= 1
              continue;
          end
          % Distances calculations
          d_ij = (position(i,1)-position(j,1))^2 + (position(i,2)-position(j,2))^2 + (position(i,3)-position(j,3))^2;
          d_ij = d_ij^0.5;
          % Set contact to be 1 if smaller than cutoff
          if d_ij <= connect_cutoff
              net(Rev_NO_aminoacid(i), Rev_NO_aminoacid(j)) = 1;
          end
      end
  end

  chain = PDB{3};
  chain = chain(:,3);
  map(:,1) = NO_aminoacid;
  map(:,2) = Rev_NO_aminoacid;
  fid1 = fopen('Outputs/1uud_mapping.txt', 'w');
  fprintf(fid1,'%s %d \t %d \n',chain{1},map(1,1),map(1,2));
  for i = 2 : number_of_atom
      if abs(NO_aminoacid(i)-NO_aminoacid(i-1)) > 0
          fprintf(fid1,'%s %d \t %d \n',chain{i},map(i,1),map(i,2));
      end
  end
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %save the contact matrix (static network)
  save ('Outputs/1uud_contact.dat','net','-ascii') 
  
end

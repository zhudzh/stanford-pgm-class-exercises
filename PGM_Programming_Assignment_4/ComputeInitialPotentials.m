%COMPUTEINITIALPOTENTIALS Sets up the cliques in the clique tree that is
%passed in as a parameter.
%
%   P = COMPUTEINITIALPOTENTIALS(C) Takes the clique tree skeleton C which is a
%   struct with three fields:
%   - nodes: cell array representing the cliques in the tree.
%   - edges: represents the adjacency matrix of the tree.
%   - factorList: represents the list of factors that were used to build
%   the tree. 
%   
%   It returns the standard form of a clique tree P that we will use through 
%   the rest of the assigment. P is struct with two fields:
%   - cliqueList: represents an array of cliques with appropriate factors 
%   from factorList assigned to each clique. Where the .val of each clique
%   is initialized to the initial potential of that clique.
%   - edges: represents the adjacency matrix of the tree. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function P = ComputeInitialPotentials(C)

% number of cliques
N = length(C.nodes);

% initialize cluster potentials 
P.cliqueList = repmat(struct('var', [], 'card', [], 'val', []), N, 1);
P.edges = zeros(N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% First, compute an assignment of factors from factorList to cliques. 
% Then use that assignment to initialize the cliques in cliqueList to 
% their initial potentials. 

% C.nodes is a list of cliques.
% So in your code, you should start with: P.cliqueList(i).var = C.nodes{i};
% Print out C to get a better understanding of its structure.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = length(C.factorList);
selected = zeros(M);
for i=1:N,
    initFactor = true;
    f = struct('var', [], 'card', [], 'val', []);
    P.cliqueList(i).var = C.nodes{i};
    for j=1:M,
        if (ismember(C.factorList(j).var, C.nodes{i}) && not(selected(j))),
            if(initFactor),
                f = C.factorList(j);
                initFactor = false;
            else
                f = FactorProduct(f, C.factorList(j));
            end
            selected(j) = 1;
        end
    end
    P.cliqueList(i) = ReorderFactorVariables(f);
end
P.edges = C.edges;
end


function out = ReorderFactorVariables(in) 
% Function accepts a factor and reorders the factor variables
% such that they are in ascending order

[S, I] = sort(in.var);

out.var = S;
out.card = in.card(I);

allAssignmentsIn = IndexToAssignment(1:prod(in.card), in.card);
allAssignmentsOut = allAssignmentsIn(:,I); % Map from in assgn to out assgn
out.val(AssignmentToIndex(allAssignmentsOut, out.card)) = in.val;

end
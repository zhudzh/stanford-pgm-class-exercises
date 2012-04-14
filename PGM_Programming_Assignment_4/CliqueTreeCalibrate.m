%CLIQUETREECALIBRATE Performs sum-product or max-product algorithm for 
%clique tree calibration.

%   P = CLIQUETREECALIBRATE(P, isMax) calibrates a given clique tree, P 
%   according to the value of isMax flag. If isMax is 1, it uses max-sum
%   message passing, otherwise uses sum-product. This function 
%   returns the clique tree where the .val for each clique in .cliqueList
%   is set to the final calibrated potentials.
%
% Copyright (C) Daphne Koller, Stanford University, 2012

function P = CliqueTreeCalibrate(P, isMax)


% Number of cliques in the tree.
N = length(P.cliqueList);

% Setting up the messages that will be passed.
% MESSAGES(i,j) represents the message going from clique i to clique j. 
MESSAGES = repmat(struct('var', [], 'card', [], 'val', []), N, N);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% We have split the coding part for this function in two chunks with
% specific comments. This will make implementation much easier.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% YOUR CODE HERE
% While there are ready cliques to pass messages between, keep passing
% messages. Use GetNextCliques to find cliques to pass messages between.
% Once you have clique i that is ready to send message to clique
% j, compute the message and put it in MESSAGES(i,j).
% Remember that you only need an upward pass and a downward pass.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(isMax),
    for i=1:N,
        P.cliqueList(i).val = log(P.cliqueList(i).val);
    end
end


while(true),
    [i, j] = GetNextCliques(P, MESSAGES);
    if (i == 0 && j == 0),
        break;
    else
        cliqueFactor = P.cliqueList(i);
        for k = 1:N,
            if (k!=j)
                if(isMax),
                    cliqueFactor = FactorSum(cliqueFactor, ...
                                             MESSAGES(k,i));
                else
                    cliqueFactor = FactorProduct(cliqueFactor,  ...
                                                 MESSAGES(k, i));
                end
            end
        end
        vari = P.cliqueList(i).var;
        varj = P.cliqueList(j).var;
        varSum = setdiff(vari, intersect(vari, varj));
        if(isMax),
            cliqueFactor = FactorMaxMarginalization(cliqueFactor, ...
                                                    varSum);
        else
            cliqueFactor = FactorMarginalization(cliqueFactor, varSum);
            cliqueFactor.val = cliqueFactor.val ./ ...
                sum(cliqueFactor.val);
        end
        MESSAGES(i,j) = cliqueFactor;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Now the clique tree has been calibrated. 
% Compute the final potentials for the cliques and place them in P.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:N,
    clFactor = P.cliqueList(i);
    msgs = MESSAGES(:,i);
    for j = 1:length(msgs),
        if(isMax),
            clFactor = FactorSum(clFactor, msgs(j));
        else 
            clFactor = FactorProduct(clFactor, msgs(j));
        end
    end
    P.cliqueList(i) = clFactor;
end
end


function C = FactorSum(A, B)

% Check for empty factors
if (isempty(A.var)), C = B; return; end;
if (isempty(B.var)), C = A; return; end;

% Check that variables in both A and B have the same cardinality
[dummy iA iB] = intersect(A.var, B.var);
if ~isempty(dummy)
	% A and B have at least 1 variable in common
	assert(all(A.card(iA) == B.card(iB)), 'Dimensionality mismatch in factors');
end

% Set the variables of C
C.var = union(A.var, B.var);

% Construct the mapping between variables in A and B and variables in C.
% In the code below, we have that
%
%   mapA(i) = j, if and only if, A.var(i) == C.var(j)
% 
% and similarly 
%
%   mapB(i) = j, if and only if, B.var(i) == C.var(j)
%
% For example, if A.var = [3 1 4], B.var = [4 5], and C.var = [1 3 4 5],
% then, mapA = [2 1 3] and mapB = [3 4]; mapA(1) = 2 because A.var(1) = 3
% and C.var(2) = 3, so A.var(1) == C.var(2).

[dummy, mapA] = ismember(A.var, C.var);
[dummy, mapB] = ismember(B.var, C.var);

% Set the cardinality of variables in C
C.card = zeros(1, length(C.var));
C.card(mapA) = A.card;
C.card(mapB) = B.card;

% Initialize the factor values of C:
%   prod(C.card) is the number of entries in C
C.val = zeros(1,prod(C.card));

% Compute some helper indices
% These will be very useful for calculating C.val
% so make sure you understand what these lines are doing.
assignments = IndexToAssignment(1:prod(C.card), C.card);
indxA = AssignmentToIndex(assignments(:, mapA), A.card);
indxB = AssignmentToIndex(assignments(:, mapB), B.card);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE:
% Correctly populate the factor values of C
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C.val = A.val(indxA) .+ B.val(indxB);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

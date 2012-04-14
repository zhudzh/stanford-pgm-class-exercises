%COMPUTEEXACTMARGINALSBP Runs exact inference and returns the marginals
%over all the variables (if isMax == 0) or the max-marginals (if isMax == 1). 
%
%   M = COMPUTEEXACTMARGINALSBP(F, E, isMax) takes a list of factors F,
%   evidence E, and a flag isMax, runs exact inference and returns the
%   final marginals for the variables in the network. If isMax is 1, then
%   it runs exact MAP inference, otherwise exact inference (sum-prod).
%   It returns an array of size equal to the number of variables in the 
%   network where M(i) represents the ith variable and M(i).val represents 
%   the marginals of the ith variable. 
%
% Copyright (C) Daphne Koller, Stanford University, 2012


function M = ComputeExactMarginalsBP(F, E, isMax)

% initialization
% you should set it to the correct value in your code
M = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%
% Implement Exact and MAP Inference.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M = repmat(struct('var', [], 'card', [], 'val', []), ...
           length(unique([F(:).var])), 1);
cTree = CliqueTreeCalibrate(CreateCliqueTree(F, E), isMax);
for i=1:length(cTree.cliqueList),
    c = cTree.cliqueList(i);
    for j = 1:length(c.var),
        if(length(M(c.var(j)).var) == 0),
            if(isMax),
                M(c.var(j)) = FactorMaxMarginalization(c, ...
                                                       setdiff(c.var,c.var(j)));
            else
                M(c.var(j)) = FactorMarginalization(c, setdiff(c.var, ...
                                                           c.var(j)));
                M(c.var(j)).val = M(c.var(j)).val ./ ...
                    sum(M(c.var(j)).val);
            end
        end
    end
end
end

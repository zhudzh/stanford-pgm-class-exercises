% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeLinearExpectations( I )
  % Inputs: An influence diagram I with a single decision node and one or more utility nodes.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  % You may assume that there is a unique optimal decision.
  %
  % This is similar to OptimizeMEU except that we will have to account for
  % multiple utility factors.  We will do this by calculating the expected
  % utility factors and combining them, then optimizing with respect to that
  % combined expected utility factor.  
  MEU = [];
  OptimalDecisionRule = [];
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE
  %
  % A decision rule for D assigns, for each joint assignment to D's parents, 
  % probability 1 to the best option from the EUF for that joint assignment 
  % to D's parents, and 0 otherwise.  Note that when D has no parents, it is
  % a degenerate case we can handle separately for convenience.
  %
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  D = I.DecisionFactors(1);
  MEU = 0;
  UF = I.UtilityFactors;
  euf = struct('var', [], 'card', [], 'val', []);
  for i = 1:length(UF),
      I.UtilityFactors = UF(i);
      euf = FactorSum(euf, CalculateExpectedUtilityFactor(I));
  end
  OptimalDecisionRule = euf;
  OptimalDecisionRule.val = zeros(1, length(euf.val));
  dIdx = find(not(euf.var .- D.var(1)));
  marked = zeros(length(euf.val));
  for i = 1:length(euf.val),
      if (not(marked(i))),
          fstA = i;
          sndA = i + 2^(dIdx - 1);
          if (euf.val(fstA) > euf.val(sndA)),
              MEU = MEU + euf.val(fstA);
              OptimalDecisionRule.val(fstA) = 1;
          else
              MEU = MEU + euf.val(sndA);
              OptimalDecisionRule.val(sndA) = 1;
          end
          marked(fstA) = 1;
          marked(sndA) = 1;
      end
  end

end

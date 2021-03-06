% Copyright (C) Daphne Koller, Stanford University, 2012

function [MEU OptimalDecisionRule] = OptimizeMEU( I )

  % Inputs: An influence diagram I with a single decision node and a single utility node.
  %         I.RandomFactors = list of factors for each random variable.  These are CPDs, with
  %              the child variable = D.var(1)
  %         I.DecisionFactors = factor for the decision node.
  %         I.UtilityFactors = list of factors representing conditional utilities.
  % Return value: the maximum expected utility of I and an optimal decision rule 
  % (represented again as a factor) that yields that expected utility.
  
  % We assume I has a single decision node.
  % You may assume that there is a unique optimal decision.
  D = I.DecisionFactors(1);

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %
  % YOUR CODE HERE...
  % 
  % Some other information that might be useful for some implementations
  % (note that there are multiple ways to implement this):
  % 1.  It is probably easiest to think of two cases - D has parents and D 
  %     has no parents.
  % 2.  You may find the Matlab/Octave function setdiff useful.
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  MEU = 0;
  euf = CalculateExpectedUtilityFactor( I );
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

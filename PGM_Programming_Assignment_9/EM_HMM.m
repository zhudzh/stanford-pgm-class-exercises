% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  P.c = sum(ClassProb) ./ sum(sum(ClassProb));
  P.clg = repmat(struct('mu_y', zeros(1,K), 'mu_x', zeros(1,K), 'mu_angle', zeros(1,K), ...
                      'sigma_y', zeros(1,K), 'sigma_x', zeros(1,K), 'sigma_angle', ...
                      zeros(1,K), 'theta', []), 1, size(poseData, 2));
  for i = 1:size(poseData,2),
      for j=1:K
          if (length(size(G)) == 3),
              hasnoparent = G(i,1,j) == 0;
              parent = G(i,2, j);
          else
              hasnoparent = G(i,1) == 0;
              parent = G(i,2);
          end
          
          if(hasnoparent),
              data = poseData;
              weights = ClassProb(:, j);
              [P.clg(i).mu_y(j), P.clg(i).sigma_y(j)] = ...
                  FitG(data(:, i, 1), weights);
              [P.clg(i).mu_x(j), P.clg(i).sigma_x(j)] = ...
                  FitG(data(:, i, 2), weights);
              [P.clg(i).mu_angle(j), P.clg(i).sigma_angle(j)] = ...
                  FitG(data(:, i, 3), weights);
    
          else
              data = poseData;
              weights = ClassProb(:, j);
              parentvalues = squeeze(data(:, parent, :));
              [beta1, P.clg(i).sigma_y(j)] = ...
                  FitLG(data(:, i, 1), parentvalues, weights);
              P.clg(i).theta(j, 1:4) = [beta1'(1,4) beta1'(1, 1:3)];
              [beta2, P.clg(i).sigma_x(j)] = ...
                  FitLG(data(:, i, 2), parentvalues, weights);
              P.clg(i).theta(j, 5:8) = [beta2'(1,4) beta2'(1,1:3)];
              [beta3, P.clg(i).sigma_angle(j)] = ...
                  FitLG(data(:, i, 3), parentvalues, weights);
              P.clg(i).theta(j, 9:12) = [beta3'(1,4) beta3'(1,1:3)];
              P.clg(i).mu_y = [];
              P.clg(i).mu_x = [];
              P.clg(i).mu_angle = [];
          end
      end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  transVector = sum(PairProb) ./ sum(sum(PairProb));
  P.transMatrix = reshape(transVector, K, K);
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for i=1:N,
      for j=1:K,
          logEmissionProb(i,j) = LogLikelihoodPerPoint(P, ...
                                                       G, ...
                                                       poseData, i, j);
      end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for l=1:L;
      factorList = [];
      for a=1:length(actionData(l).marg_ind)
          if(a==1)
              init = struct('var', [1], ...
                            'card', [K], 'val', log(P.c));
              initEmission = BuildEmissionFactor(a, actionData(l).marg_ind(a), ...
                                                 logEmissionProb, K);
              factorList = [init initEmission];
          else
              tran = BuildTransFactor(a, ...
                                      P.transMatrix, ...
                                      K);
              emission = BuildEmissionFactor(a, actionData(l).marg_ind(a), ...
                                                 logEmissionProb, K);
              factorList = [factorList tran emission];
          end
      end
      [M, pCali] = ComputeExactMarginalsHMM(factorList);
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);

end

function f = BuildEmissionFactor(a, ind, logEmission, K)
f = struct('var', [a], 'card', [K], 'val', logEmission(ind, :));
end

function f = BuildTransFactor(a, transMatrix, K)
f = struct('var', [a-1, a], 'card', [K, K], 'val', log(reshape(transMatrix, ...
                                                  1, K^2)));
end

function ll = LogLikelihoodPerPoint(P, G, dataset, d, label)
ll = 0;
for i = 1:size(dataset,2),
    if (length(size(G)) == 3),
        hasnoparent = G(i,1,label) == 0;
        parent = G(i,2, label);
    else
        hasnoparent = G(i,1) == 0;
        parent = G(i,2);
    end
    if(hasnoparent),
        ll += lognormpdf(dataset(d,i,1), P.clg(i).mu_y(label), ...
                         P.clg(i).sigma_y(label));
        ll += lognormpdf(dataset(d,i,2), P.clg(i).mu_x(label), ...
                         P.clg(i).sigma_x(label));
        ll += lognormpdf(dataset(d,i,3), P.clg(i).mu_angle(label), ...
                         P.clg(i).sigma_angle(label));
    else
        paparams = [1, dataset(d,parent,1),dataset(d,parent,2),dataset(d,parent,3)];
        mu_y = P.clg(i).theta(label, 1:4) * paparams';
        mu_x = P.clg(i).theta(label, 5:8) * paparams';
        mu_angle = P.clg(i).theta(label, 9:12) * paparams';
        ll += lognormpdf(dataset(d,i,1), mu_y, ...
                         P.clg(i).sigma_y(label));
        ll += lognormpdf(dataset(d,i,2), mu_x, ...
                         P.clg(i).sigma_x(label));
        ll += lognormpdf(dataset(d,i,3), mu_angle, ...
                         P.clg(i).sigma_angle(label));
    end
end
end

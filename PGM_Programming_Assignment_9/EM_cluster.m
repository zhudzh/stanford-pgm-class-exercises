% File: EM_cluster.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb] = EM_cluster(poseData, G, InitialClassProb, maxIter)

% INPUTS
% poseData: N x 10 x 3 matrix, where N is number of poses;
%   poseData(i,:,:) yields the 10x3 matrix for pose i.
% G: graph parameterization as explained in PA8
% InitialClassProb: N x K, initial allocation of the N poses to the K
%   classes. InitialClassProb(i,j) is the probability that example i belongs
%   to class j
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K, conditional class probability of the N examples to the
%   K classes in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to class j

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);

ClassProb = InitialClassProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  %
  % Fill in P.c with the estimates for prior class probabilities
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  %
  % Hint: This part should be similar to your work from PA8
  
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
  
  % E-STEP to re-estimate ClassProb using the new parameters
  %
  % Update ClassProb with the new conditional class probabilities.
  % Recall that ClassProb(i,j) is the probability that example i belongs to
  % class j.
  %
  % You should compute everything in log space, and only convert to
  % probability space at the end.
  %
  % Tip: To make things faster, try to reduce the number of calls to
  % lognormpdf, and inline the function (i.e., copy the lognormpdf code
  % into this file)
  %
  % Hint: You should use the logsumexp() function here to do
  % probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  logPrior = log(P.c);
  ll = zeros(N,1);
  for i=1:N,
      for j=1:K,
          ClassProb(i,j) = LogLikelihoodPerPoint(P, ...
                                                 G, ...
                                                 poseData, i, j);
      end
      ClassProb(i, :) = ClassProb(i, :) + logPrior;
      ll(i,1) = logsumexp(ClassProb(i,:));
      ClassProb(i, :) = exp(ClassProb(i,:) .- ll(i,1));
  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Compute log likelihood of poseData for this iteration
  % Hint: You should use the logsumexp() function here
  loglikelihood(iter) = 0;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loglikelihood(iter) = sum(sum(ll));
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting: when loglikelihood decreases
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
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

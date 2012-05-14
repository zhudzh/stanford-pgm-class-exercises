function [P loglikelihood] = LearnCPDsGivenGraph(dataset, G, labels)
%
% Inputs:
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% G: graph parameterization as explained in PA description
% labels: N x 2 true class labels for the examples. labels(i,j)=1 if the 
%         the ith example belongs to class j and 0 elsewhere        
%
% Outputs:
% P: struct array parameters (explained in PA description)
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset, 1);
K = size(labels,2);

loglikelihood = 0;
P.c = zeros(1,K);

% estimate parameters
% fill in P.c, MLE for class probabilities
% fill in P.clg for each body part and each class
% choose the right parameterization based on G(i,1)
% compute the likelihood - you may want to use ComputeLogLikelihood.m
% you just implemented.
%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P.c = sum(labels) ./ sum(sum(labels));
P.clg = repmat(struct('mu_y', zeros(1,K), 'mu_x', zeros(1,K), 'mu_angle', zeros(1,K), ...
                      'sigma_y', zeros(1,K), 'sigma_x', zeros(1,K), 'sigma_angle', ...
                      zeros(1,K), 'theta', []), 1, size(dataset, 2));
for i = 1:size(dataset,2),
    for j=1:K
        if (length(size(G)) == 3),
        hasparent = G(i,1,j) == 0;
        parent = G(i,2, j);
    else
        hasparent = G(i,1) == 0;
        parent = G(i,2);
    end

        if(hasparent),
            data = dataset(find(labels(:, j)), :, :);
            [P.clg(i).mu_y(j), P.clg(i).sigma_y(j)] = ...
                FitGaussianParameters(data(:, i, 1));
            [P.clg(i).mu_x(j), P.clg(i).sigma_x(j)] = ...
                FitGaussianParameters(data(:, i, 2));
            [P.clg(i).mu_angle(j), P.clg(i).sigma_angle(j)] = ...
                FitGaussianParameters(data(:, i, 3));
    
        else
            data = dataset(find(labels(:, j)), :, :);
            parentvalues = squeeze(data(:, parent, :));
            [beta1, P.clg(i).sigma_y(j)] = ...
                FitLinearGaussianParameters(data(:, i, 1), ...
                                            parentvalues);
            P.clg(i).theta(j, 1:4) = [beta1'(1,4) beta1'(1, 1:3)];
            [beta2, P.clg(i).sigma_x(j)] = ...
                FitLinearGaussianParameters(data(:, i, 2), ...
                                            parentvalues);
            P.clg(i).theta(j, 5:8) = [beta2'(1,4) beta2'(1,1:3)];
            [beta3, P.clg(i).sigma_angle(j)] = ...
                FitLinearGaussianParameters(data(:, i, 3), ...
                                            parentvalues);
            P.clg(i).theta(j, 9:12) = [beta3'(1,4) beta3'(1,1:3)];
            P.clg(i).mu_y = [];
            P.clg(i).mu_x = [];
            P.clg(i).mu_angle = [];
        end
    end
end
loglikelihood = ComputeLogLikelihood(P, G, dataset);

fprintf('log likelihood: %f\n', loglikelihood);


function loglikelihood = ComputeLogLikelihood(P, G, dataset)
% returns the (natural) log-likelihood of data given the model and graph structure
%
% Inputs:
% P: struct array parameters (explained in PA description)
% G: graph structure and parameterization (explained in PA description)
%
%    NOTICE that G could be either 10x2 (same graph shared by all classes)
%    or 10x2x2 (each class has its own graph). your code should compute
%    the log-likelihood using the right graph.
%
% dataset: N x 10 x 3, N poses represented by 10 parts in (y, x, alpha)
% 
% Output:
% loglikelihood: log-likelihood of the data (scalar)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

N = size(dataset,1); % number of examples
K = length(P.c); % number of classes

loglikelihood = 0;
% You should compute the log likelihood of data as in eq. (12) and (13)
% in the PA description
% Hint: Use lognormpdf instead of log(normpdf) to prevent underflow.
%       You may use log(sum(exp(logProb))) to do addition in the original
%       space, sum(Prob).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
llGivenK = zeros(1, K);
logPrior = log(P.c);
for d = 1:length(dataset),
    ll = zeros(1, K);
    for label = 1:K,
        gaussList = GaussianList(P, G, label);
        ll(1, label) = logPrior(label) + LogLikelihoodPerPoint(gaussList, ...
                                                          dataset(d,:,:));
    end
    %    ll
    log(sum(exp(ll)))
    loglikelihood = loglikelihood + log(sum(exp(ll)));
end
% for label = 1:K,
%     ll = zeros(1, length(dataset));
%     gaussList = GaussianList(P, G, label);
%     for d = 1:length(dataset),
%         ll(1, d) = LogLikelihoodPerPoint(gaussList, dataset(d,:,:));
%     end
%     llGivenK(1, label) = sum(ll)
% end
% logPrior = log(P.c);
% llMarg = log(sum(exp(logPrior + llGivenK)));

%gaussList = GuassianList(P, G, 1);
% GaussianList(P,G,1)(4).g
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function ll = LogLikelihoodPerPoint(gaussList, d)
ll = 0;
for i = 1:size(d,2),
    for j = 1:size(d,3),
        newll = lognormpdf(d(1, i, j), ...
                           gaussList(i).g(j, 1), ...
                           gaussList(i).g(j, 2));
        ll = ll + newll;
    end
end
end

function gaussList = GaussianList(P, G, k)
    P_copy = P;
    gaussList = repmat(struct('g', []), 1, length(P_copy.clg));
    for i=1:length(P_copy.clg),
        [gauss, P_copy] = Gaussian(P_copy,G,i,k);
        gaussList(1, i).g = gauss;
    end
end

function [gauss, P_copy] = Gaussian(P_copy, G, i, k)
    parent = G(i,2);
    if (G(i,1) == 0)
        gauss = [P_copy.clg(i).mu_y(k), P_copy.clg(i).sigma_y(k);...
                P_copy.clg(i).mu_x(k), P_copy.clg(i).sigma_x(k);...
                P_copy.clg(i).mu_angle(k), P_copy.clg(i).sigma_angle(k)];
    else
        paparams = [1, P_copy.clg(parent).mu_y(k), P_copy.clg(parent).mu_x(k), ...
                    P_copy.clg(parent).mu_angle(k)];
        mu_y = P_copy.clg(i).theta(k, 1:4) * paparams';
        mu_x = P_copy.clg(i).theta(k, 5:8) * paparams';
        mu_angle = P_copy.clg(i).theta(k, 9:12) * paparams';
        P_copy.clg(i).mu_y(k) = mu_y;
        P_copy.clg(i).mu_x(k) = mu_x;
        P_copy.clg(i).mu_angle(k) = mu_angle;
        gauss = [mu_y, P_copy.clg(i).sigma_y(k);...
                mu_x, P_copy.clg(i).sigma_x(k);...
                mu_angle, P_copy.clg(i).sigma_angle(k)];
    end
end

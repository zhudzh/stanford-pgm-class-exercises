function [loglikelihood, llperlabel] = ComputeLogLikelihood(P, G, dataset)
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
llperlabel = zeros(N, K);
logPrior = log(P.c);
for d = 1:length(dataset),
    ll = zeros(1, K);
    for label = 1:K,
        ll(1, label) = logPrior(label) + LogLikelihoodPerPoint(P,G, ...
                                                          dataset,d,label);
    end
    llperlabel(d, :) = ll;
    loglikelihood = loglikelihood + log(sum(exp(ll)));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function ll = LogLikelihoodPerPoint(P, G, dataset, d, label)
ll = 0;
for i = 1:size(dataset,2),
    if (length(size(G)) == 3),
        hasparent = G(i,1,label) == 0;
        parent = G(i,2, label);
    else
        hasparent = G(i,1) == 0;
        parent = G(i,2);
    end
    if(hasparent),
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


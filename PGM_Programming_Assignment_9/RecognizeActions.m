% File: RecognizeActions.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [accuracy, predicted_labels] = RecognizeActions(datasetTrain, datasetTest, G, maxIter)

% INPUTS
% datasetTrain: dataset for training models, see PA for details
% datasetTest: dataset for testing models, see PA for details
% G: graph parameterization as explained in PA decription
% maxIter: max number of iterations to run for EM

% OUTPUTS
% accuracy: recognition accuracy, defined as (#correctly classified examples / #total examples)
% predicted_labels: N x 1 vector with the predicted labels for each of the instances in datasetTest, with N being the number of unknown test instances


% Train a model for each action
% Note that all actions share the same graph parameterization and number of max iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
models = repmat(struct('c',[],'clg',[],'transMatrix',[]), ...
                1, length(datasetTrain));
for i=1:length(datasetTrain),
    [P, ll, class, pair] = EM_HMM(datasetTrain(i).actionData, ...
                                  datasetTrain(i).poseData, ...
                                  G, datasetTrain(i).InitialClassProb, ...
                                  datasetTrain(i).InitialPairProb, ...
                                  maxIter);
    models(i) = P;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Classify each of the instances in datasetTrain
% Compute and return the predicted labels and accuracy
% Accuracy is defined as (#correctly classified examples / #total examples)
% Note that all actions share the same graph parameterization

accuracy = 0;
predicted_labels = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% YOUR CODE HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = length(models);
N = length(datasetTest.poseData);
logEPs = repmat(struct('logEmissionProb', zeros(N,K)), 1, K);
predicted_labels = zeros(length(datasetTest.actionData),1);
for m=1:length(models),
    P=models(m);
    logEmissionProb = zeros(N,K);
    for i=1:N,
        for j=1:K,
            logEmissionProb(i,j) = LogLikelihoodPerPoint(P, ...
                                                         G, ...
                                                         datasetTest.poseData, i, j);
        end
    end
    logEPs(1, m).logEmissionProb = logEmissionProb;
end
    
for act_ind=1:length(datasetTest.actionData),
    actionData = datasetTest.actionData;
    ll = zeros(1, K);
    for m=1:length(models),
        P=models(m);
        logEmissionProb = logEPs(1,m).logEmissionProb;
        loglikelihood = 0;
        factorList = [];
        for a=1:length(actionData(act_ind).marg_ind)
            if(a==1)
                init = struct('var', [1], ...
                              'card', [K], 'val', log(P.c));
                initEmission = BuildEmissionFactor(a, actionData(act_ind).marg_ind(a), ...
                                                   logEmissionProb, K);
                factorList = [init initEmission];
            else
                tran = BuildTransFactor(a, ...
                                        P.transMatrix, ...
                                        K);
                emission = BuildEmissionFactor(a, actionData(act_ind).marg_ind(a), ...
                                               logEmissionProb, K);
                factorList = [factorList tran emission];
            end
        end
        [X, pCali] = ComputeExactMarginalsHMM(factorList);
        ll(1,m) = logsumexp(pCali.cliqueList(1).val);
    end
    [v,i] = max(ll);
    predicted_labels(act_ind, 1) = i;
    disp(sprintf('EM iteration %d: log likelihood: %f', ...
                 act_ind, i));
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end

end
result = (predicted_labels == datasetTest.labels);
accuracy = sum(result)/length(result);
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function n = NormalizeFromLog(logValues)
s = logsumexp(logValues);
n = exp(logValues .- s);
end

function n = normalize(v)
n = v ./ sum(v);
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

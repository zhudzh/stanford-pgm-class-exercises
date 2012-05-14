% function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)
% returns the negative log-likelihood and its gradient, given a CRF with parameters theta,
% on data (X, y). 
%
% Inputs:
% X            Data.                           (numCharacters x numImageFeatures matrix)
%              X(:,1) is all ones, i.e., it encodes the intercept/bias term.
% y            Data labels.                    (numCharacters x 1 vector)
% theta        CRF weights/parameters.         (numParams x 1 vector)
%              These are shared among the various singleton / pairwise features.
% modelParams  Struct with three fields:
%   .numHiddenStates     in our case, set to 26 (26 possible characters)
%   .numObservedStates   in our case, set to 2  (each pixel is either on or off)
%   .lambda              the regularization parameter lambda
%
% Outputs:
% nll          Negative log-likelihood of the data.    (scalar)
% grad         Gradient of nll with respect to theta   (numParams x 1 vector)
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [nll, grad] = InstanceNegLogLikelihood(X, y, theta, modelParams)

    % featureSet is a struct with two fields:
    %    .numParams - the number of parameters in the CRF (this is not numImageFeatures
    %                 nor numFeatures, because of parameter sharing)
    %    .features  - an array comprising the features in the CRF.
    %
    % Each feature is a binary indicator variable, represented by a struct 
    % with three fields:
    %    .var          - a vector containing the variables in the scope of this feature
    %    .assignment   - the assignment that this indicator variable corresponds to
    %    .paramIdx     - the index in theta that this feature corresponds to
    %
    % For example, if we have:
    %   
    %   feature = struct('var', [2 3], 'assignment', [5 6], 'paramIdx', 8);
    %
    % then feature is an indicator function over X_2 and X_3, which takes on a value of 1
    % if X_2 = 5 and X_3 = 6 (which would be 'e' and 'f'), and 0 otherwise. 
    % Its contribution to the log-likelihood would be theta(8) if it's 1, and 0 otherwise.
    %
    % If you're interested in the implementation details of CRFs, 
    % feel free to read through GenerateAllFeatures.m and the functions it calls!
    % For the purposes of this assignment, though, you don't
    % have to understand how this code works. (It's complicated.)
    
    featureSet = GenerateAllFeatures(X, modelParams);

    % Use the featureSet to calculate nll and grad.
    % This is the main part of the assignment, and it is very tricky - be careful!
    % You might want to code up your own numerical gradient checker to make sure
    % your answers are correct.
    %
    % Hint: you can use CliqueTreeCalibrate to calculate logZ effectively. 
    %       We have halfway-modified CliqueTreeCalibrate; complete our implementation 
    %       if you want to use it to compute logZ.
    
    nll = 0;
    grad = zeros(size(theta));
    %%%
    % Your code here:
    % construct clique tree
    ctree = ConstructCliqueTree(X, y, theta, featureSet, ...
                                modelParams);
    [caltree, logZ] = CliqueTreeCalibrate(ctree, false);
    featureCnt = FeatureCount(y, theta, ...
                                      featureSet, modelParams);
    weightedFeatureCount = theta * featureCnt';
    regCost = (modelParams.lambda/2)*(theta * theta');
    nll = logZ - weightedFeatureCount + regCost;
    
    exptFC = ModelExptFeatureCount (caltree, featureSet, theta);
    regGrad = modelParams.lambda .* theta;
    grad = exptFC - featureCnt + regGrad;
end

function exptFC = ModelExptFeatureCount (caltree, featureSet, theta)
    exptFC = zeros(1, length(theta));
    margs = repmat(struct('var', [], 'card', [], 'val', []), 1, ...
                   length(caltree.cliqueList) + 1);
    for i = 1:length(caltree.cliqueList),
        cli = caltree.cliqueList(i);
        caltree.cliqueList(i).val = cli.val ./ ...
            sum(cli.val);
        margs(i) = FactorMarginalization(cli, cli.var(2));
        margs(i).val = margs(i).val ./ sum(margs(i).val);
    end
    cli = caltree.cliqueList(length(caltree.cliqueList));
    margs(length(caltree.cliqueList)+1) = FactorMarginalization(cli, ...
                                                      cli.var(1));
    margs(length(caltree.cliqueList)+1).val = ...
        margs(length(caltree.cliqueList)+1).val ./ ...
        sum(margs(length(caltree.cliqueList)+1).val);
    
    for f=1:length(featureSet.features),
        fea = featureSet.features(f);
        if(length(fea.var) == 1),
            prob = GetValueOfAssignment(margs(fea.var), ...
                                        fea.assignment);
            exptFC(fea.paramIdx) = exptFC(fea.paramIdx) + prob;
        else
            prob = GetValueOfAssignment(caltree.cliqueList(fea.var(1)), ...
                                        fea.assignment);
            exptFC(fea.paramIdx) = exptFC(fea.paramIdx) + prob;
        end
    end
end

function featureCnt = FeatureCount(y, theta, ...
                                   featureSet, ...
                                   modelParams)
    featureCnt = zeros(1, length(theta));
    modelExptFeatureCnt= zeros(1, length(theta));
    for f=1:length(featureSet.features),
        fea = featureSet.features(f);
        if(y(fea.var) == fea.assignment),
            featureCnt(fea.paramIdx) = featureCnt(fea.paramIdx) + 1;
        end
    end
end

function ctree = ConstructCliqueTree(X, y, theta, featureSet, modelParams)
    numVars = length(y);
    singleFactors = repmat(struct('var', [], 'card', [], 'val', []), 1, numVars);
    pairFactors = repmat(struct('var', [], 'card', [], 'val', []), 1, numVars-1);
    card = modelParams.numHiddenStates;
    for v = 1:numVars,
        singleFactors(v) = struct('var', [v], 'card', ...
                               [card], 'val', ...
                                  ones(1, prod([card])));
        if (v<numVars),
            pairFactors(v) = struct('var', [v, v+1], 'card', [card, ...
                                card], 'val', ones(1, prod([card, ...
                                card])));
        end
    end
    for f=1:length(featureSet.features),
        fea = featureSet.features(f);
        if(length(fea.var)==1),
            fact = singleFactors(fea.var(1));
            cur = GetValueOfAssignment(fact, fea.assignment);
            singleFactors(fea.var(1)) = SetValueOfAssignment(fact, ...
                                                             fea.assignment, cur * exp(theta(fea.paramIdx)));
        else
            fact = pairFactors(fea.var(1));
            cur = GetValueOfAssignment(fact, fea.assignment);
            pairFactors(fea.var(1)) = SetValueOfAssignment(fact, ...
                                                           fea.assignment, cur * exp(theta(fea.paramIdx)));
        end
    end
    ctree = CreateCliqueTree([singleFactors pairFactors]);
end
function [valid_labels] = run_knn(k, train_data, train_labels, valid_data)
% knn_predict: Uses the supplied training inputs and labels to make
%              predictions for validation data using the K-nearest neighbours
%              algorithm.
%
% Note: N_TRAIN is the number of training examples,
%       N_VALID is the number of validation examples, 
%       and M is the number of features per example.
%
% Inputs:
%   k:            The number of neighbours to use for classification 
%                 of a validation example.
%   train_data:   The N_TRAIN x M matrix of training
%                 data.
%   train_labels: The N_TRAIN x 1 vector of training labels
%                 corresponding to the examples in train_data 
%                 (must be binary).
%   valid_data:    The N_VALID x M matrix of data to
%                 predict classes for.
%
% Outputs:
%   valid_labels: The N_VALID x 1 vector of predicted labels for the validation data.
%

narginchk(4,4);

dist = l2_distance(valid_data', train_data');
[sorted_dist, nearest] = sort(dist,2);

nearest = nearest(:,1:k);
valid_labels = train_labels(nearest);

% For binary labels, return 0/1 based on a threshold of 0.5
%valid_labels = mean(valid_labels,2) >= 0.5;

% For other cases, use the mode to find the most frequent predicted label
% for each validation case 
valid_labels = mode(valid_labels, 2);

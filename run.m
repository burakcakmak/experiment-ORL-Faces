%% Load data
subjects = 40;
samples = 10;
pixels = prod([112 92]); % Image dimension: 112 x 92
X = zeros(subjects, samples, pixels);
Y = zeros(subjects, samples, 1);
for i = 1:subjects
    for j = 1:samples
        tmp = imread(strcat('orl_faces/s', int2str(i), '/', int2str(j), '.pgm'));
        tmp = reshape(tmp.', 1, pixels);
        X(i, j, :) = tmp;
        Y(i, j, :) = i;
    end
end


%% Define data sets
% Define training set
tmp = randperm(10,3);
train_X = X(:, ~ismember([1:10], tmp), :); % drop the 10th sample from training
train_Y = Y(:, ~ismember([1:10], tmp), :);
% Define test set
test_X = X(:, tmp, :); % use the 10th sample for testing
test_Y = Y(:, tmp, :);


%% Reshape data in appropriate sizes to feed the classifier
convert_size = @(x) reshape(x, size(x, 1) * size(x, 2), size(x, 3));
train_X = convert_size(train_X);
train_Y = convert_size(train_Y);
test_X = convert_size(test_X);
test_Y = convert_size(test_Y);
X = convert_size(X);
Y = convert_size(Y);


%% Examine quiality of MATLAB's KNN classifier (ClassificationKNN)
fprintf('Examining quality of the MATLAB''s KNN classifier. Define: \n');
fprintf(['  Resubstitution loss: by default, this is the fraction ', ...
    'of misclassifications from the predictions of model.\n']);
fprintf(['  Cross-validation loss: this is the average loss of each cross-', ...
    'validation model when predicting on data that is not used for training.\n\n']);

% Test using euclidean distance and different number of neighbours
for NumNeighbors = 2:5
    fprintf('Model: euclidean distance, %d neighbours\n', NumNeighbors);
    
    % Build model
    model = ClassificationKNN.fit(X, Y, ... % Using all data for this model
        'NumNeighbors', NumNeighbors, ...
        'Distance','cosine'); 
    % Create a cross-validated classifier from the model. 
    cv_model = crossval(model);
    
    % Stats 
    fprintf('\tResubstitution loss: %.2f%%.\n', 100 * resubLoss(model));
    fprintf('\tCross-validation loss: %.2f%%.\n', 100 * kfoldLoss(cv_model));
end

% Test using euclidean distance and 5 neighbours
NumNeighbors = 1;
fprintf('Model: euclidean distance, %d neighbours\n', NumNeighbors);
% Build model
model = ClassificationKNN.fit(train_X, train_Y, ... 
    'NumNeighbors', NumNeighbors, ...
    'Distance','cosine'); 
% Test 
fprintf('\tClassification rate on test data: %.2f\n', 100*(1-numel(find(test_Y ~= predict(model, test_X)))/numel(test_Y)));

%% Test another implementation 
fprintf('\n\nExamining quality of the another classifier that uses l2_distance function.\n');
for NumNeighbors = 1:5
    fprintf('Model: euclidean, %d neighbours\n', NumNeighbors);
    predict_labels = knn(NumNeighbors, train_X, train_Y, test_X);
    fprintf('\tClassification rate on test data: %.2f\n', 100*(1-numel(find(test_Y ~= predict_labels))/numel(test_Y)));
end

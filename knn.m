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
train_X = X(:, [1:8], :); % drop the 10th sample from training
train_Y = Y(:, [1:8], :);
% Define test set
test_X = X(:, [9, 10], :); % use the 10th sample for testing
test_Y = Y(:, [9, 10], :);

%% Reshape data in appropriate sizes to feed the classifier
convert_size = @(x) reshape(x, size(x, 1) * size(x, 2), size(x, 3));
train_X = convert_size(train_X);
train_Y = convert_size(train_Y);
test_X = convert_size(test_X);
test_Y = convert_size(test_Y);

%% Build model
NumNeighbors = 5;
model = ClassificationKNN.fit(train_X, train_Y, 'NumNeighbors', NumNeighbors);

loss(model, test_X, test_Y, 'lossfun', 'classiferror')
fprintf('Examine the Quality of the KNN Classifier\n')
fprintf('The classifier predicts incorrectly for %.2f%% of the training data.\n', 100*resubLoss(model));
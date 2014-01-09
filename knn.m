%% Load data
subjects = 40;
samples = 10;
img_dim = [112 92];
pixels = img_dim(1) * img_dim(2);
X = zeros(subjects, samples, pixels);
Y = zeros(subjects, samples, 1);
for i = 1:subjects
    for j = 1:samples
        tmp = imread(strcat('orl_faces/s', int2str(i), '/', int2str(j), '.pgm'));
        tmp = reshape(tmp, 1, pixels);
        X(i, j, :) = tmp;
        Y(i, j, :) = i;
    end
end

%% Define train and test sets
train_X = X(:, 1:9, :); % drop the 10th sample from training
train_Y = Y(:, 1:9, :);
test_X = X(:, 10, :); % use the 10th sample for testing
test_Y = Y(:, 10, :);

%% Build model

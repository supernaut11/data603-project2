%% Load data
data = load('pose.mat');
pose = data.pose;
[d1,d2,num_poses,num_subjects] = size(pose);

%% Divide data into test and training
num_training = 10;
num_testing = num_poses - num_training;

training_data = zeros(d1, d2, 1, num_subjects * num_training);

for subj = 1:num_subjects
    start_idx = (subj-1)*num_training;
    for p = 1:num_training
        training_data(:,:,1,start_idx+p) = pose(:,:,p,subj);
    end
end

training_labels = categorical((kron(1:num_subjects, ones(1, num_training)))');

testing_data = zeros(d1, d2, 1, num_subjects * num_testing);

for subj = 1:num_subjects
    start_idx = (subj-1)*num_testing;
    for p = 1:num_testing
        testing_data(:,:,1,start_idx+p) = pose(:,:,num_training+p,subj);
    end
end

testing_labels = categorical((kron(1:num_subjects, ones(1, num_testing)))');

%% Construct the neural network architecture
% Very poor performance, does not converge
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,20)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% reasonably good accuracy, roughly around 60 percent
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,16,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% Worse performance than first neural network (changed convolution layer
% values)
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,12,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% % AlexNet - 0.6127
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2,'Stride',2)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% % 1- 0.7059
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1,'DilationFactor',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
%     maxPooling2dLayer(3)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% 2- 0.7304
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1,'DilationFactor',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
%     averagePooling2dLayer(3)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% 3- accuracy 0.7255 (when dilationfactor gone and averagepoolinglayer(5))
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1,'DilationFactor',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
%     averagePooling2dLayer(5)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% % 4- 0.7500 accuracy
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1,'DilationFactor',1)
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
%     averagePooling2dLayer(5)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(num_subjects)
%     softmaxLayer
%     classificationLayer];

% % 0.7696!! 0.7794 when learning rate is 1e-2
layers = [...
    imageInputLayer([d1,d2,1])
    convolution2dLayer(5,16,'Padding',1,'DilationFactor',1)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3)
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2)
    groupedConvolution2dLayer(3,24,2,'Padding','same')
    reluLayer
    batchNormalizationLayer
%     maxPooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding',1)
%     reluLayer
%     groupedConvolution2dLayer(3,36,2,'Padding','same')
%     reluLayer
%     groupedConvolution2dLayer(3,36,2,'Padding','same')
%     reluLayer
    averagePooling2dLayer(4)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
    fullyConnectedLayer(512)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(num_subjects)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs', 60, ...
    'InitialLearnRate', 1e-2, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(training_data, training_labels, layers, options);

%% Classify images using trained neural network
testing_labels_predicted = classify(net, testing_data);
accuracy = sum(testing_labels_predicted == testing_labels) / numel(testing_labels);

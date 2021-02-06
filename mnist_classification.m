%% Load data
data = load('mnist.mat');
training_images = data.imgs_train;
testing_images = data.imgs_test;
training_labels = data.labels_train;
testing_labels = data.labels_test;

[d1,d2,num_train] = size(training_images);
nn_training_images = zeros(d1, d2, 1, num_train);

for idx = 1:num_train
    nn_training_images(:,:,1,idx) = training_images(:,:,idx);
end

[d1,d2,num_test] = size(testing_images);
nn_testing_images = zeros(d1, d2, 1, num_test);

for idx = 1:num_test
    nn_testing_images(:,:,1,idx) = testing_images(:,:,idx);
end

%% Show some images
figure;
colormap gray;
perm = randperm(num_test,20);
for idx = 1:20
    subplot(4,5,idx);
    imagesc(testing_images(:,:,perm(idx)));
end

% %% Train the neural network
% % 1- 0.9440 accuracy
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,20)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];
% 
% % 2- 0.9774 accuracy
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,10,'Padding',1)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 3- 0.9236 accuracy
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,10,'Padding',1,'Stride',3)
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 4- 0.9808 accuracy
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,10,'DilationFactor',[1 1],'Padding','same')
%     clippedReluLayer(10)
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 5- 0.8971
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(3,10,'DilationFactor',[1 1],'Padding',1)
%     sigmoidLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 6- 0.9830
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'DilationFactor',[1 1],'Padding','same')
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 7- 0.9825
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'DilationFactor',1,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3,'Stride',2)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 8- 0.9877!!!
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'DilationFactor',1,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% % 0.9846
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'DilationFactor',1,'Padding','same')
%     reluLayer
%     batchNormalizationLayer
%     maxPooling2dLayer(3)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

% AlexNet variant from class
% layers = [...
%     imageInputLayer([d1,d2,1])
%     convolution2dLayer(5,16,'Padding',1)
%     reluLayer
%     batchNormalizationLayer
% %     crossChannelNormalizationLayer(5)
%     maxPooling2dLayer(2,'Stride',2)
%     groupedConvolution2dLayer(3,24,2,'Padding','same')
%     reluLayer
% %     crossChannelNormalizationLayer(5)
% %     maxPooling2dLayer(2,'Stride',2)
% %     convolution2dLayer(3,32,'Padding',1)
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
% %     reluLayer
% %     groupedConvolution2dLayer(3,36,2,'Padding','same')
%     reluLayer
%     maxPooling2dLayer(2,'Stride',2)
% %     fullyConnectedLayer(512)
% %     reluLayer
% %     dropoutLayer(0.5)
%     fullyConnectedLayer(512)
%     reluLayer
%     dropoutLayer(0.5)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 1e-3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(nn_training_images, training_labels, layers, options);

%% Evaluate test data using trained network
testing_labels_predicted = classify(net, nn_testing_images);
accuracy = sum(testing_labels_predicted == testing_labels) / numel(testing_labels);

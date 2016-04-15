%!/usr/bin/matlab
% Neural network Example, XOR

%% configuration
trainset_size = 500;
evalset_size = 20;
hidden_sizes = [20];

%% populate dataset
input  = (randn(2, trainset_size)-0.5).*2; % input \in [-1,+1]
target = -prod(sign(input), 1);
input_eval = (randn(2, evalset_size)-0.5).*2;
target_eval = -prod(sign(input_eval), 1);

%% initialize network
net = feedforwardnet(hidden_sizes);
net.trainParam.showWindow = 1;

%% train
net = train(net, input, target);

%% evaluate
output_eval = sim(net, input_eval);
error_eval = abs(sign(output_eval) - target_eval);
disp(error_eval);

%!/usr/bin/matlab
% example on Iris Dataset

disp('FIXME: this example still does not work, but why?')

%% read original dataset
iris_raw = textread('../dataset/iris/iris');
iris_raw_feature = iris_raw(:, 1:4);
iris_raw_target  = iris_raw(:, 5);

%% data pre-process and split
data_input_train = iris_raw_feature(...
    [1:40, 51:90, 101:140], :)';
data_target_train = iris_raw_target(...
    [1:40, 51:90, 101:140], :)';
data_input_test = iris_raw_feature(...
    [41:50, 91:100, 141:150], :)';
data_target_test = iris_raw_target(...
    [41:50, 91:100, 141:150], :)';

%% initialize network
size_hidden = 20;
net = patternnet(size_hidden);
view(net)
%net.trainParam.goal = 0

%% train network
net = train(net, data_input_train, data_target_train)
view(net);

%% evaluate
eval_test = net(data_input_test);
perf = perform(net, data_target_test, eval_test)
classes = vec2ind(eval_test)

disp(abs(eval_test - data_target_test));

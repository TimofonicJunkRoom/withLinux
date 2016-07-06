% nn test, 3 layer

% init data, y = x
data = {};
data.train = {};
data.test  = {};
data.train.N = 10;
data.train.feature = rand(1, data.train.N);
data.train.label   = data.train.feature;
data.test.N = 10;
data.test.feature  = rand(1, data.test.N);
data.test.label    = data.test.feature;
% done init data

% init weight
W1 = randn(20, 1)/2;
B1 = randn(20, 1)/2;
W2 = randn(1, 20)/2;
B2 = randn(1, 1)/2;
% done init weight

% basic configuration
maxiter = 1000;
lr = 0.0001;
sigmoid = @(x)(1./(1-exp(x)));
crit    = @(gt,out)(sum((abs(gt-out).^2)/2));
% done config

% train with GD
for iter = 1:maxiter
   % forward pass
   I = data.train.feature;
   h = W1 * I + repmat(B1, 1, data.train.N);
   H = sigmoid(h);
   o = W2 * H + repmat(B2, 1, data.train.N);
   E = crit(data.train.label, o);
   disp(sprintf('iter %d loss %f', iter, E));
   if isnan(E)
      break
   end
   if iter > 2
      if abs(E - prevE) < 0.001
         break
      end
   end
   prevE = E;

   % backward pass
   delta2 = data.train.label - o; % [1,100]
   dEbW2 = delta2 * H.'; % [1,100]*[20,100]'->[1,20]
   dEbB2 = delta2 * ones(data.train.N, 1); % [1,100]*[100,1]->[1,1]

   delta1 = W2.' * delta2 .*(H).*(1-H); %[20,1]*[1,100]->[20,100]
   dEbW1 = delta1 * I.'; % [20,100]*[1,100]'->[20,1]
   dEbB1 = delta1 * ones(data.train.N, 1); % [20,100]*[100,1]->[20,1]

   % update
   W2 = W2 + lr * dEbW2;
   B2 = B2 + lr * dEbB2;
   W1 = W1 + lr * dEbW1;
   B1 = B1 + lr * dEbB1;
end
% done train

% perform test
I = data.train.feature;
h = W1 * I + repmat(B1, 1, data.train.N);
H = sigmoid(h);
o = W2 * H + repmat(B2, 1, data.train.N);
E = crit(data.train.label, o);
disp(sprintf('test loss %f', E));
% done test

% graph
figure
plot(I, 'r');
hold on;
plot(o, 'b');
% done graph

if E < 2
  pause;
end

%!/usr/bin/octave
% 3-layer neural network

%%
disp('I: preparing data');
size_data = [1, 1000]; 
X = rand(size_data(1),size_data(2));
T = X; % f: x |-> x
X(:,1)
T(:,1)

%%
disp('I: 3-layer neural network configurations');
lr = 0.001; % learning rate
max_iter = 200; % max iter
m = 1; % size(X_i) = m*1
h = 1; % size(XH_i) = h*1
c = 1; % size(XY_i) = c*1

%%
disp('I: initializing network parameter');
WH = rand(h, m); WH
WY = rand(c, h); WY
BH = rand(h, 1); BH
BY = rand(c, 1); BY
WH0 = WH;
WY0 = WY;
BH0 = BH;
BY0 = BY;

%%
disp('I: setting up utility functions');
loss = @(x, t)( (norm(t - x)^2)/2 )%;
dtanh = @(x)(1./(cos(x).^2))%;
forward = @(x, WH, BH, WY, BY)( WY * tanh(WH * x + BH) + BY )

%%
disp('I: setting up storage stack');
stack_loss = [];
stack_loss0 = [];

%%
iter = 0;
data_cursor = 1;
for iter = 1:max_iter
  %disp('   fetch data');
  if data_cursor > size_data(2)
    data_cursor = 1;
  end
  x = X(:,data_cursor); % fetch input
  t = T(:,data_cursor); % fetch label
  data_cursor = data_cursor + 1;

  disp(sprintf('I: iter %d', iter));

  %disp('   forward');
  uH = WH * x + BH; %size(uH)
  xH = tanh(uH); %size(xH)
  uY = WY * xH + BY; %size(uY)
  xY = uY; %tanh(uY); %size(xY)
  l = loss(xY, t);
  stack_loss = [ stack_loss, l ];
  stack_loss0 = [ stack_loss0, loss(forward(x, WH0, BH0, WY0, BY0), t) ];
  disp(sprintf('   loss %f', l));

  %disp('   backward');
  gbY = (xY - t) .* dtanh(uY); %size(gbY)% delta Y
  gbH = (WY' * gbY) .* dtanh(uH); %size(gbH)% delta H
  gwY = gbY * xH'; %size(gwY)
  gwH = gbH * x'; %size(gwH)

  %disp('   update');
  BY = BY - (lr .* gbY);
  WY = WY - (lr .* gwY);
  BH = BH - (lr .* gbH);
  WH = WH - (lr .* gwH);

  %disp('   update stat');
  sum(sum(lr .* gbY))
end

%%
disp('I: dump param');
WY
WH
BY
BH

%%
disp('I: validate');
forward([ 1], WH, BH, WY, BY)
forward([ 0.5], WH, BH, WY, BY)

%% draw graph
figure;
grid on;
hold on;
plot(stack_loss);
plot(stack_loss0, 'r');
print -dpdf stack_loss.pdf;

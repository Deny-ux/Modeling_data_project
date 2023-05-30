format long;
clear all;
close all;

% Load data
static_data = load('danestat50.txt');
u = static_data(:,1);
y = static_data(:, 2);

u_training = u(1:2:length(u));
y_training = y(1:2:length(y));

u_validation = u(2:2:length(u));
y_validation = y(2:2:length(y));

% Find the coefficients for approximate function
M = horzcat(u_training, ones(length(u_training), 1));
W = M\y_training;

% Create the string function representation
functionString = ['@(x) ', num2str(W(2)), ' + ', num2str(W(1)), '*x'];

% Convert string function representation to function
f = str2func(functionString);

% Display aproximation function with training data
figure;
fplot(f, [-1, 1], 'red'); 
hold on;
scatter(u_training, y_training, 'blue');
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Statyczny model liniowy na tle danych uczących')
legend('model liniowy', 'dane uczące')
hold off;

% Display linear model with validation data
figure;
fplot(f, [-1, 1], 'red'); 
hold on;
scatter(u_validation, y_validation, 'blue');
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Statyczny model liniowy na tle danych weryfikujących')
legend('model liniowy', 'dane weryfikujące')
hold off;

% Find mean square error using predefined function
% For training data
y_training_appr = f(u_training);
rmse_training_data = find_MSE(y_training, y_training_appr)

% For validation data
y_validation_appr = f(y_validation);
rmse_validation_data = find_MSE(y_validation, y_training_appr)




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
% u = linspace(-1, 1, 100);


%% N = 2

% Dane uczące
M2 = horzcat(u_training.^2, u_training, ones(length(u_training), 1));
W2 = M2\y_training;
functionString2 = ['@(x) ', num2str(W2(3)), ' + ', num2str(W2(2)), '*x', ' + ', num2str(W2(1)), '*x^2'];

f2 = str2func(functionString2);

figure;
plot_model_with_data(f2, u_training, y_training, ...
    'Statyczny model nieliniowy na tle danych uczących, N = 2', ...
    'Model', 'Dane uczące')

rmse_n2_ucz = find_MSE(y_training, polyval(W2, u_training))

% Dane weryfikujące
figure;
plot_model_with_data(f2, u_validation, y_validation, ...
    'Statyczny model nieliniowy na tle danych weryfikujących, N = 2', ...
    'Model', 'Dane weryfikujące')

rmse_n2_weryf = find_MSE(y_validation, polyval(W2, u_validation))



%% N = 3

% Dane uczące
M3 = horzcat(u_training.^3, u_training.^2, u_training, ones(length(u_training), 1));
W3 = M3\y_training;
functionString3 = ['@(x) ', num2str(W3(4)), ' + ', num2str(W3(3)), '*x', ' + ', num2str(W3(2)), '*x^2', ' + ', num2str(W3(1)), '*x^3'];

f3 = str2func(functionString3);

% polyFunc3 = polyval(W3, u);
figure;
plot_model_with_data(f3, u_training, y_training, ...
    'Statyczny model nieliniowy na tle danych uczących, N = 3', ...
    'Model', 'Dane uczące')

rmse_n3_ucz = find_MSE(y_training, polyval(W3, u_training))

% Dane weryfikujące
figure;
plot_model_with_data(f3, u_validation, y_validation, ...
    'Statyczny model nieliniowy na tle danych weryfikujących, N = 3', ...
    'Model', 'Dane weryfikujące')

rmse_n3_weryf = find_MSE(y_validation, polyval(W3, u_validation))



%% N = 4

% Dane uczące
M4 = horzcat(u_training.^4, u_training.^3, u_training.^2, u_training, ones(length(u_training), 1));
W4 = M4\y_training;

functionString4 = ['@(x) ', num2str(W4(5)), ' + ', num2str(W4(4)), '*x', ' + ', num2str(W4(3)), '*x^2', ' + ', num2str(W4(2)), '*x^3', ' + ', num2str(W4(1)), '*x^4'];
f4 = str2func(functionString4);

figure;
plot_model_with_data(f4, u_training, y_training, ...
    'Statyczny model nieliniowy na tle danych uczących, N = 4', ...
    'Model', 'Dane uczące')

rmse_n4_ucz = find_MSE(y_training, polyval(W4, u_training))

% Dane weryfikujące
figure;
plot_model_with_data(f4, u_validation, y_validation, ...
    'Statyczny model nieliniowy na tle danych weryfikujących, N = 4', ...
    'Model', 'Dane weryfikujące')

rmse_n4_weryf = find_MSE(y_validation, polyval(W4, u_validation))

%% N = 5

% Dane uczące
M5 = horzcat(u_training.^5, u_training.^4, u_training.^3, u_training.^2, u_training, ones(length(u_training), 1));
W5 = M5\y_training;
functionString5 = ['@(x) ', num2str(W5(6)), ' + ', num2str(W5(5)), '*x', ' + ', num2str(W5(4)), '*x^2', ' + ', num2str(W5(3)), '*x^3', ' + ', num2str(W5(2)), '*x^4', ' + ', num2str(W5(1)), '*x^5'];
f5 = str2func(functionString5);
figure;
plot_model_with_data(f5, u_training, y_training, ...
    'Statyczny model nieliniowy na tle danych uczących, N = 5', ...
    'Model', 'Dane uczące')

rmse_n5_ucz = find_MSE(y_training, polyval(W5, u_training))

% Dane weryfikujące
figure;
plot_model_with_data(f5, u_validation, y_validation, ...
    'Statyczny model nieliniowy na tle danych weryfikujących, N = 5', ...
    'Model', 'Dane weryfikujące')

rmse_n5_weryf = find_MSE(y_validation, polyval(W5, u_validation))


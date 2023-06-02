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
w = M\y_training;


f = @(u) w(2) + w(1)*u;

plot_static_model(f, u_training, y_training, u_validation, y_validation, 1)

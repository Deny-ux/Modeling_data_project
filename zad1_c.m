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

%% N1

M = horzcat(ones(length(u_training), 1), u_training);
w1 = M\y_training;
f1 = @(u) w1(1) + w1(2)*u;
plot_static_model(f1, u_training, y_training, u_validation, y_validation, 1)
%% N2

M = horzcat(ones(length(u_training), 1), u_training, u_training.^2);
w2 = M\y_training;
f2 = @(u) w2(1) + w2(2)*u + w2(3)*u^2;
plot_static_model(f2, u_training, y_training, u_validation, y_validation, 2)

%% N3

M = horzcat(ones(length(u_training), 1), u_training, u_training.^2, u_training.^3);
w3 = M\y_training;
f3 = @(u) w3(1) + w3(2)*u + w3(3)*u^2 + w3(4)*u^3;
plot_static_model(f3, u_training, y_training, u_validation, y_validation, 3)

%% N4

M = horzcat(ones(length(u_training), 1), u_training, u_training.^2, u_training.^3, u_training.^4);
w4 = M\y_training;
f4 = @(u) w4(1) + w4(2)*u + w4(3)*u^2 + w4(4)*u^3 + w4(5)*u^4;
plot_static_model(f4, u_training, y_training, u_validation, y_validation, 4)

%% N5

M = horzcat(ones(length(u_training), 1), u_training, u_training.^2, u_training.^3, u_training.^4, u_training.^5);
w5 = M\y_training;
f5 = @(u) w5(1) + w5(2)*u + w5(3)*u^2 + w5(4)*u^3 + w5(5)*u^4 + w5(6)*u^5;
plot_static_model(f5, u_training, y_training, u_validation, y_validation, 5)

%% N6

M = horzcat(ones(length(u_training), 1), u_training, u_training.^2, u_training.^3, u_training.^4, u_training.^5, u_training.^6);
w6 = M\y_training;
f6 = @(u) w6(1) + w6(2)*u + w6(3)*u^2 + w6(4)*u^3 + w6(5)*u^4 + w6(6)*u^5 + w6(7)*u^6;
plot_static_model(f6, u_training, y_training, u_validation, y_validation, 6)



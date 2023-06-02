format long;
clear all;
close all;

dynamic_data_training = load('danedynucz50.txt');
u_training = dynamic_data_training(:, 1);
y_training = dynamic_data_training(:, 2);
k = linspace(1, 2000, 2000);
dynamic_data_validation = load('danedynwer50.txt');
u_validation = dynamic_data_validation(:, 1);
y_validation = dynamic_data_validation(:, 2);

k_start = 10;
k_used = k(1, k_start:end)';
length_k = length(k) - k_start + 1;

M_o1_d4 = horzcat( ...
    u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-1:end-1).^3, u_training(k_start-1:end-1).^4, ...
    y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-1:end-1).^3, y_training(k_start-1:end-1).^4);

w_o1_d4 = M_o1_d4\y_training(k_start:end);

ff = @(u, y) w_o1_d4(1)*u + w_o1_d4(2)*u^2 + w_o1_d4(3)*u^3 + w_o1_d4(4)*u^4 + ...
    w_o1_d4(5)*y + w_o1_d4(6)*y^2 + w_o1_d4(7)*y^3 + w_o1_d4(8)*y^4;

uu = linspace(-1, 1, 201)';
yy = zeros(201, 1);
for i = 1:201
    y_temp = zeros(400,1);
    for j = 2:2000
        y_temp(j) = ff(uu(i), y_temp(j-1));
    end
    yy(i) = y_temp(2000);
end

plot(uu, yy);
title('Charakterystyka y(u)');
xlabel('u')
ylabel('y')
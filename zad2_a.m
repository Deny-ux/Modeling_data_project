format long;
clear all;
close all;

dynamic_data_training = load('danedynucz50.txt');
u_training = dynamic_data_training(:, 1);
y_training = dynamic_data_training(:, 2);
k = linspace(1, 2000, 2000)
dynamic_data_validation = load('danedynwer50.txt');
u_validation = dynamic_data_validation(:, 1);
y_validation = dynamic_data_validation(:, 2);

figure;
plot(k, y_training);
xlabel('Numer próbki, k' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane dynamiczne uczące')
figure;
plot(k, y_validation);
xlabel('Numer próbki, k' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane dynamiczne weryfikujące')
% plot3(u_training, y_training, k)
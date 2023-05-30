format long;
clear all;
close all;

static_data = load('danestat50.txt');
u = static_data(:,1);
y = static_data(:, 2);
size = 10;
figure;
scatter(u, y, size,'MarkerEdgeColor',[0 .7 .5], ...
    'MarkerFaceColor',[0 .9 .5], ...
    'LineWidth',1);
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane statyczne')

u_training = u(1:2:length(u));
y_training = y(1:2:length(y));

u_validation = u(2:2:length(u));
y_validation = y(2:2:length(y));
figure;
scatter(u_training, y_training, 'blue');
hold on;
scatter(u_validation ,y_validation, 'red')
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane statyczne podzielone na uczące i weryfikujące')
legend('Dane uczące', 'Dane weryfikujące')
hold off;

figure;
scatter(u_training, y_training);
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane statyczne uczące')

figure;
scatter(u_validation, y_validation);
xlabel('Sygnał wejścwiowy, u' )
ylabel('Sygnał wyjścwiowy, y')
title('Dane statyczne weryfikujące')


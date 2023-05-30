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
length_k = length(k) - k_start;

Y_n1 = y_training(k_start:end, 1);

u_training_used_n1 = u_training(k_start-1:end-1, 1);
y_training_used_n1 = y_training(k_start-1:end-1, 1);

u_validation_used_n1 = u_validation(k_start-1:end-1, 1);
y_validation_used_n1 = y_validation(k_start-1:end-1, 1);

Y_n2 = y_training(k_start-1:end-1, 1);
u_training_used_n2 = u_training(k_start-2:end-2, 1);
y_training_used_n2 = y_training(k_start-2:end-2, 1);

u_validation_used_n2 = u_validation(k_start-2:end-2, 1);
y_validation_used_n2 = y_validation(k_start-2:end-2, 1);

u_training_used_n3 = u_training(k_start-3:end-3, 1);
y_training_used_n3 = y_training(k_start-3:end-3, 1);

u_validation_used_n3 = u_validation(k_start-3:end-3, 1);
y_validation_used_n3 = y_validation(k_start-3:end-3, 1);

%% Pierwszy rząd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane uczące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_n1_arx = horzcat(u_training_used_n1, y_training_used_n1);

w_n1 = M_n1_arx\Y_n1;

% bez rekurencji
y_mod_n1_arx_train = zeros(2000 - k_start + 1, 1);
for i = 1:length(y_mod_n1_arx_train)
    y_mod_n1_arx_train(i) = w_n1(1)*u_training_used_n1(i) + w_n1(2)*y_training_used_n1(i);
end

% z rekurencją
y_mod_n1_oe_train = zeros(2000 - k_start + 1, 1);
y_mod_n1_oe_train(1) = y_mod_n1_arx_train(1);
for i = 2:length(y_mod_n1_oe_train)
    y_mod_n1_oe_train(i) = w_n1(1)*u_training_used_n1(i-1) + w_n1(2)*y_mod_n1_oe_train(i-1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane weryfikujące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_mod_n1_arx_valid = zeros(2000 - k_start + 1, 1);

for i = 1:length(y_mod_n1_arx_valid)
    y_mod_n1_arx_valid(i) = w_n1(1)*u_validation_used_n1(i) + w_n1(2)*y_validation_used_n1(i);
end


% z rekurencją
y_mod_n1_oe_valid = zeros(2000 - k_start + 1, 1);
y_mod_n1_oe_valid(1) = y_mod_n1_arx_valid(1);
for i = 2:length(y_mod_n1_oe_valid)
    y_mod_n1_oe_valid(i) = w_n1(1)*u_validation_used_n1(i-1) + w_n1(2)*y_mod_n1_oe_valid(i-1);
end

plot_model_with_data(k(1, k_start:end), y_mod_n1_arx_train, y_mod_n1_oe_train, y_training_used_n1, y_mod_n1_arx_valid, y_mod_n1_oe_valid, y_validation_used_n1)



%% Drugi rząd
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane uczące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bez rekurencji
% M_n2_arx = horzcat(u_training_used_n2, y_training_used_n2);
% 
% w_n2 = M_n2_arx\Y_n2;
% 
% 
% y_mod_n1_arx_train = zeros(2000 - k_start + 1, 1);
% for i = 1:length(y_mod_n1_arx_train)
%     y_mod_n1_arx_train(i) = w_n1(1)*u_training_used_n1(i) + w_n1(2)*y_training_used_n1(i);
% end
% 
% % z rekurencją
% y_mod_n1_oe_train = zeros(2000 - k_start + 1, 1);
% y_mod_n1_oe_train(1) = y_mod_n1_arx_train(1);
% for i = 2:length(y_mod_n1_oe_train)
%     y_mod_n1_oe_train(i) = w_n1(1)*u_training_used_n1(i-1) + w_n1(2)*y_mod_n1_oe_train(i-1);
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % dane weryfikujące
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% y_mod_n1_arx_valid = zeros(2000 - k_start + 1, 1);
% 
% for i = 1:length(y_mod_n1_arx_valid)
%     y_mod_n1_arx_valid(i) = w_n1(1)*u_validation_used_n1(i) + w_n1(2)*y_validation_used_n1(i);
% end
% 
% 
% % z rekurencją
% y_mod_n1_oe_valid = zeros(2000 - k_start + 1, 1);
% y_mod_n1_oe_valid(1) = y_mod_n1_arx_valid(1);
% for i = 2:length(y_mod_n1_oe_valid)
%     y_mod_n1_oe_valid(i) = w_n1(1)*u_validation_used_n1(i-1) + w_n1(2)*y_mod_n1_oe_valid(i-1);
% end
% 
% plot_model_with_data(k(1, k_start:end), y_mod_n1_arx_train, y_mod_n1_oe_train, y_training_used_n1, y_mod_n1_arx_valid, y_mod_n1_oe_valid, y_validation_used_n1)






%% Trzeci rząd

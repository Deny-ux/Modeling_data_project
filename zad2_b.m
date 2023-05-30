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


%% Pierwszy rząd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane uczące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_n1_arx = horzcat(u_training(k_start-1:end-1), y_training(k_start-1:end-1));

w_n1 = M_n1_arx\y_training(k_start:end);

% bez rekurencji
y_mod_n1_arx_train = zeros(length_k, 1);
for i = 1:length_k
    y_mod_n1_arx_train(i) = w_n1(1)*u_training(k_start-1+i) + w_n1(2)*y_training(k_start-1+i);
end

% z rekurencją
y_mod_n1_oe_train = zeros(length_k, 1);
y_mod_n1_oe_train(1) = y_mod_n1_arx_train(1);
for i = 2:length_k
    y_mod_n1_oe_train(i) = w_n1(1)*u_training(k_start-1+i) + w_n1(2)*y_mod_n1_oe_train(i - 1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane weryfikujące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_mod_n1_arx_valid = zeros(length_k, 1);

for i = 1:length_k
    y_mod_n1_arx_valid(i) = w_n1(1)*u_validation(k_start -1 + i) + w_n1(2)*y_validation(k_start -1 + i);
end


% z rekurencją
y_mod_n1_oe_valid = zeros(length_k, 1);
y_mod_n1_oe_valid(1) = y_mod_n1_arx_valid(1);
for i = 2:length_k
    y_mod_n1_oe_valid(i) = w_n1(1)*u_validation(k_start + i-1) + w_n1(2)*y_mod_n1_oe_valid(i-1);
end

plot_model_with_data(k_used, y_mod_n1_arx_train, y_mod_n1_oe_train,  ...
    y_training(k_start:end), y_mod_n1_arx_valid, y_mod_n1_oe_valid, y_validation(k_start:end), 1)



%% Drugi rząd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane uczące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_n2 = horzcat(u_training(k_start-1:end-1), u_training(k_start-2:end-2), ...
    y_training(k_start-1:end-1), y_training(k_start-2:end-2));

w_n2 = M_n2\y_training(k_start:end);


% bez rekurencji
y_mod_n2_arx_train = zeros(length_k, 1);
for i = 1:length_k
    y_mod_n2_arx_train(i) = w_n2(1)*u_training(k_start-1+i) + w_n2(2)*u_training(k_start-2+i) ...
        + w_n2(3)*y_training(k_start-1+i) + w_n2(4)*y_training(k_start-2+i);
end

% z rekurencją
y_mod_n2_oe_train = zeros(length_k, 1);
y_mod_n2_oe_train(1) = y_mod_n2_arx_train(1);
y_mod_n2_oe_train(2) = y_mod_n2_arx_train(2);
for i = 3:length_k
    y_mod_n2_oe_train(i) = w_n2(1)*u_training(k_start-1+i) + w_n2(2)*u_training(k_start-2+i) ...
        + w_n2(3)*y_mod_n2_oe_train(i - 1) + w_n2(4)*y_mod_n2_oe_train(i - 2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane weryfikujące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bez rekurencji
y_mod_n2_arx_valid = zeros(length_k, 1);

for i = 1:length_k
    y_mod_n2_arx_valid(i) = w_n2(1)*u_validation(k_start-1+i) + w_n2(2)*u_validation(k_start-2+i) ...
        + w_n2(3)*y_validation(k_start-1+i) + w_n2(4)*y_validation(k_start-2+i);
end

% z rekurencją
y_mod_n2_oe_valid = zeros(length_k, 1);
y_mod_n2_oe_valid(1) = y_mod_n2_arx_valid(1);
y_mod_n2_oe_valid(2) = y_mod_n2_arx_valid(2);

for i = 3:length_k
    y_mod_n2_oe_valid(i) = w_n2(1)*u_validation(k_start-1+i) + w_n2(2)*u_validation(k_start-2+i) ...
        + w_n2(3)*y_mod_n2_oe_valid(i - 1) + w_n2(4)*y_mod_n2_oe_valid(i -2);
end

plot_model_with_data(k_used, y_mod_n2_arx_train, y_mod_n2_oe_train,  ...
    y_training(k_start:end), y_mod_n2_arx_valid, y_mod_n2_oe_valid, y_validation(k_start:end), 2)

%% Trzeci rząd

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane uczące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M_n3 = horzcat(u_training(k_start-1:end-1), u_training(k_start-2:end-2), u_training(k_start-3:end-3), ...
    y_training(k_start-1:end-1), y_training(k_start-2:end-2), y_training(k_start-3:end-3));

w_n3 = M_n3\y_training(k_start:end);

% bez rekurencji
y_mod_n3_arx_train = zeros(length_k, 1);
for i = 1:length_k
    y_mod_n3_arx_train(i) = w_n3(1)*u_training(k_start-1+i) + w_n3(2)*u_training(k_start-2+i) ...
        + w_n3(3)*u_training(k_start-3+i) + w_n3(4)*y_training(k_start-1+i) + w_n3(5)*y_training(k_start-2+i) ...
        + w_n3(6)*y_training(k_start-3+i);
end

% z rekurencją
y_mod_n3_oe_train = zeros(length_k, 1);
y_mod_n3_oe_train(1) = y_mod_n3_arx_train(1);
y_mod_n3_oe_train(2) = y_mod_n3_arx_train(2);
y_mod_n3_oe_train(3) = y_mod_n3_arx_train(3);
for i = 4:length_k
    y_mod_n3_oe_train(i) = w_n3(1)*u_training(k_start-1+i) + w_n3(2)*u_training(k_start-2+i) ...
        + w_n3(3)*u_training(k_start-3+i) + w_n3(4)*y_mod_n3_oe_train(i - 1) + w_n3(5)*y_mod_n3_oe_train(i - 2)...
        + w_n3(6)*y_mod_n3_oe_train(i - 3);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dane weryfikujące
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_mod_n3_arx_valid = zeros(length_k, 1);

for i = 1:length_k
    y_mod_n3_arx_valid(i) = w_n3(1)*u_validation(k_start-1+i) + w_n3(2)*u_validation(k_start-2+i) ...
        + w_n3(3)*u_validation(k_start-3+i) + w_n3(4)*y_validation(k_start-1+i) + w_n3(5)*y_validation(k_start-2+i) ...
        + w_n3(6)*y_validation(k_start-3+i);
end

% z rekurencją
y_mod_n3_oe_valid = zeros(length_k, 1);
y_mod_n3_oe_valid(1) = y_mod_n3_arx_valid(1);
y_mod_n3_oe_valid(2) = y_mod_n3_arx_valid(2);
y_mod_n3_oe_valid(3) = y_mod_n3_arx_valid(3);

for i = 4:length_k
    y_mod_n3_oe_valid(i) = w_n3(1)*u_validation(k_start-1+i) + w_n3(2)*u_validation(k_start-2+i) ...
        + w_n3(3)*u_validation(k_start-3+i) + w_n3(4)*y_mod_n3_oe_valid(i - 1) + w_n3(5)*y_mod_n3_oe_valid(i -2) ...
        + w_n3(6)*y_mod_n3_oe_valid(i -3);
end

plot_model_with_data(k_used, y_mod_n3_arx_train, y_mod_n3_oe_train,  ...
    y_training(k_start:end), y_mod_n3_arx_valid, y_mod_n3_oe_valid, y_validation(k_start:end), 3)



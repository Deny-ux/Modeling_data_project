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

order_dynamics = 2;
polynomial_degree = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1, 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 1 && polynomial_degree == 2
    

    M_o1_d2 = horzcat(u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2);

    w_o1_d2 = M_o1_d2\y_training(k_start:end);
    f_o1_d2 = @(u, y, w) w(1)*u + w(2)*(u)^2 + w(3)*y + w(4)*(y)^2;
    
    %  dane uczące
    
    % bez rekurencji
    y_mod_o1_d2_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d2_arx_train(i) = f_o1_d2(u_training(k_start-2+i), y_training(k_start-2+i), w_o1_d2);
    end

    % z rekurencją
    y_mod_o1_d2_oe_train = zeros(length_k, 1);
    y_mod_o1_d2_oe_train(1) = y_mod_o1_d2_arx_train(1);
    for i = 2:length_k
        y_mod_o1_d2_oe_train(i) =  f_o1_d2(u_training(k_start-2+i), y_mod_o1_d2_oe_train(i - 1), w_o1_d2);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o1_d2_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d2_arx_valid(i) = f_o1_d2(u_validation(k_start-2+i), y_validation(k_start-2+i), w_o1_d2);
    end

    % z rekurencją
    y_mod_o1_d2_oe_valid = zeros(length_k, 1);
    y_mod_o1_d2_oe_valid(1) = y_mod_o1_d2_arx_valid(1);
    for i = 2:length_k
        y_mod_o1_d2_oe_valid(i) = f_o1_d2(u_validation(k_start-2+i), y_mod_o1_d2_oe_valid(i-1), w_o1_d2);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o1_d2_arx_train, y_mod_o1_d2_oe_train, y_training(k_start:end), ...
        y_mod_o1_d2_arx_valid, y_mod_o1_d2_oe_valid, y_validation(k_start:end), 1, 2)  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1, 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 1 && polynomial_degree == 3

    M_o1_d3 = horzcat( ...
        u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-1:end-1).^3, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-1:end-1).^3);

    w_o1_d3 = M_o1_d3\y_training(k_start:end);
    f_o1_d3 = @(u, y, w) w(1)*u + w(2)*(u)^2 + w(3)*(u)^3 + w(4)*(y) + w(5)*(y)^2 + w(6)*(y)^3;
    
    %  dane uczące
    
    % bez rekurencji
    y_mod_o1_d3_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d3_arx_train(i) = f_o1_d3(u_training(k_start-2+i), y_training(k_start-2+i), w_o1_d3);
    end

    % z rekurencją
    y_mod_o1_d3_oe_train = zeros(length_k, 1);
    y_mod_o1_d3_oe_train(1) = y_mod_o1_d3_arx_train(1);
    
    for i = 2:length_k
        y_mod_o1_d3_oe_train(i) =  f_o1_d3(u_training(k_start-2+i), y_mod_o1_d3_oe_train(i - 1), w_o1_d3);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o1_d3_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d3_arx_valid(i) = f_o1_d3(u_validation(k_start-2+i), y_validation(k_start-2+i), w_o1_d3);
    end

    % z rekurencją
    y_mod_o1_d3_oe_valid = zeros(length_k, 1);
    y_mod_o1_d3_oe_valid(1) = y_mod_o1_d3_arx_valid(1);
    for i = 2:length_k
        y_mod_o1_d3_oe_valid(i) = f_o1_d3(u_validation(k_start-2+i), y_mod_o1_d3_oe_valid(i-1), w_o1_d3);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o1_d3_arx_train, y_mod_o1_d3_oe_train, y_training(k_start:end), ...
        y_mod_o1_d3_arx_valid, y_mod_o1_d3_oe_valid, y_validation(k_start:end), 1, 3)  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1, 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 1 && polynomial_degree == 4

    M_o1_d4 = horzcat( ...
        u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-1:end-1).^3, u_training(k_start-1:end-1).^4, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-1:end-1).^3, y_training(k_start-1:end-1).^4);

    w_o1_d4 = M_o1_d4\y_training(k_start:end);
    f_o1_d4 = @(u, y, w) w(1)*u + w(2)*(u)^2 + w(3)*(u)^3 + + w(4)*(u)^4+ w(5)*(y) + w(6)*(y)^2 + w(7)*(y)^3 + w(8)*(y)^4;

    %  dane uczące

    % bez rekurencji
    y_mod_o1_d4_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d4_arx_train(i) = f_o1_d4(u_training(k_start-2+i), y_training(k_start-2+i), w_o1_d4);
    end

    % z rekurencją
    y_mod_o1_d4_oe_train = zeros(length_k, 1);
    y_mod_o1_d4_oe_train(1) = y_mod_o1_d4_arx_train(1);

    for i = 2:length_k
        y_mod_o1_d4_oe_train(i) =  f_o1_d4(u_training(k_start-2+i), y_mod_o1_d4_oe_train(i - 1), w_o1_d4);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o1_d4_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o1_d4_arx_valid(i) = f_o1_d4(u_validation(k_start-2+i), y_validation(k_start-2+i), w_o1_d4);
    end

    % z rekurencją
    y_mod_o1_d4_oe_valid = zeros(length_k, 1);
    y_mod_o1_d4_oe_valid(1) = y_mod_o1_d4_arx_valid(1);
    for i = 2:length_k
        y_mod_o1_d4_oe_valid(i) = f_o1_d4(u_validation(k_start-2+i), y_mod_o1_d4_oe_valid(i-1), w_o1_d4);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o1_d4_arx_train, y_mod_o1_d4_oe_train, y_training(k_start:end), ...
        y_mod_o1_d4_arx_valid, y_mod_o1_d4_oe_valid, y_validation(k_start:end), 1, 4)  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2, 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 2 && polynomial_degree == 2

    M_o2_d2 = horzcat( ...
        u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-2:end-2), u_training(k_start-2:end-2).^2, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-2:end-2), y_training(k_start-2:end-2).^2);

    w_o2_d2 = M_o2_d2\y_training(k_start:end);
    f_o2_d2 = @(u_minus_1, u_minus_2, y_minus_1, y_minus_2, w) w(1)*u_minus_1 + w(2)*(u_minus_1)^2 + w(3)*(u_minus_2) + w(4)*(u_minus_2)^2 + w(5)*(y_minus_1) + w(6)*(y_minus_1)^2 + w(7)*(y_minus_2) + w(8)*(y_minus_2)^2;

    %  dane uczące

    % bez rekurencji
    y_mod_o2_d2_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d2_arx_train(i) = f_o2_d2(u_training(k_start-2+i), u_training(k_start-3+i), y_training(k_start-2+i),  y_training(k_start-3+i), w_o2_d2);
    end

    % z rekurencją
    y_mod_o2_d2_oe_train = zeros(length_k, 1);
    y_mod_o2_d2_oe_train(1) = y_mod_o2_d2_arx_train(1);
    y_mod_o2_d2_oe_train(2) = y_mod_o2_d2_arx_train(2);
    for i = 3:length_k
        y_mod_o2_d2_oe_train(i) =  f_o2_d2(u_training(k_start-2+i), u_training(k_start-3+i), y_mod_o2_d2_oe_train(i - 1), y_mod_o2_d2_oe_train(i - 2), w_o2_d2);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o2_d2_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d2_arx_valid(i) = f_o2_d2(u_validation(k_start-2+i), u_validation(k_start-3+i), y_validation(k_start-2+i),  y_validation(k_start-3+i), w_o2_d2);
    end

    % z rekurencją
    y_mod_o2_d2_oe_valid = zeros(length_k, 1);
    y_mod_o2_d2_oe_valid(1) = y_mod_o2_d2_arx_valid(1);
    y_mod_o2_d2_oe_valid(2) = y_mod_o2_d2_arx_valid(2);

    for i = 3:length_k
        y_mod_o2_d2_oe_valid(i) = f_o2_d2(u_validation(k_start-2+i), u_validation(k_start-3+i), y_mod_o2_d2_oe_valid(i - 1), y_mod_o2_d2_oe_valid(i - 2), w_o2_d2);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o2_d2_arx_train, y_mod_o2_d2_oe_train, y_training(k_start:end), ...
        y_mod_o2_d2_arx_valid, y_mod_o2_d2_oe_valid, y_validation(k_start:end), 2, 2)  
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2, 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 2 && polynomial_degree == 3

    M_o2_d3 = horzcat( ...
        u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-1:end-1).^3, u_training(k_start-2:end-2), u_training(k_start-2:end-2).^2, u_training(k_start-2:end-2).^3, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-1:end-1).^3, y_training(k_start-2:end-2), y_training(k_start-2:end-2).^2, y_training(k_start-2:end-2).^3);

    w_o2_d3 = M_o2_d3\y_training(k_start:end);
    f_o2_d3 = @(u_minus_1, u_minus_2, y_minus_1, y_minus_2, w) w(1)*u_minus_1 + w(2)*(u_minus_1)^2 + w(3)*(u_minus_1)^3 + w(4)*(u_minus_2) + w(5)*(u_minus_2)^2 +w(6)*(u_minus_2)^3 + w(7)*(y_minus_1)+ w(8)*(y_minus_1)^2 + + w(9)*(y_minus_1)^3 + w(10)*(y_minus_2)+ w(11)*(y_minus_2)^2 + + w(12)*(y_minus_2)^3; 

    %  dane uczące

    % bez rekurencji
    y_mod_o2_d3_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d3_arx_train(i) = f_o2_d3(u_training(k_start-2+i), u_training(k_start-3+i), y_training(k_start-2+i),  y_training(k_start-3+i), w_o2_d3);
    end

    % z rekurencją
    y_mod_o2_d3_oe_train = zeros(length_k, 1);
    y_mod_o2_d3_oe_train(1) = y_mod_o2_d3_arx_train(1);
    y_mod_o2_d3_oe_train(2) = y_mod_o2_d3_arx_train(2);
    for i = 3:length_k
        y_mod_o2_d3_oe_train(i) =  f_o2_d3(u_training(k_start-2+i), u_training(k_start-3+i), y_mod_o2_d3_oe_train(i - 1), y_mod_o2_d3_oe_train(i - 2), w_o2_d3);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o2_d3_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d3_arx_valid(i) = f_o2_d3(u_validation(k_start-2+i), u_validation(k_start-3+i), y_validation(k_start-2+i),  y_validation(k_start-3+i), w_o2_d3);
    end

    % z rekurencją
    y_mod_o2_d3_oe_valid = zeros(length_k, 1);
    y_mod_o2_d3_oe_valid(1) = y_mod_o2_d3_arx_valid(1);
    y_mod_o2_d3_oe_valid(2) = y_mod_o2_d3_arx_valid(2);
    for i = 3:length_k
        y_mod_o2_d3_oe_valid(i) = f_o2_d3(u_validation(k_start-2+i), u_validation(k_start-3+i), y_mod_o2_d3_oe_valid(i - 1), y_mod_o2_d3_oe_valid(i - 2), w_o2_d3);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o2_d3_arx_train, y_mod_o2_d3_oe_train, y_training(k_start:end), ...
        y_mod_o2_d3_arx_valid, y_mod_o2_d3_oe_valid, y_validation(k_start:end), 2, 3)  
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2, 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if order_dynamics == 2 && polynomial_degree == 4

    M_o2_d4 = horzcat( ...
        u_training(k_start-1:end-1), u_training(k_start-1:end-1).^2, u_training(k_start-1:end-1).^3, u_training(k_start-1:end-1).^4, u_training(k_start-2:end-2), u_training(k_start-2:end-2).^2, u_training(k_start-2:end-2).^3, u_training(k_start-2:end-2).^4, ...
        y_training(k_start-1:end-1), y_training(k_start-1:end-1).^2, y_training(k_start-1:end-1).^3, y_training(k_start-1:end-1).^4, y_training(k_start-2:end-2), y_training(k_start-2:end-2).^2, y_training(k_start-2:end-2).^3, y_training(k_start-2:end-2).^4);

    w_o2_d4 = M_o2_d4\y_training(k_start:end);
    f_o2_d4 = @(u_minus_1, u_minus_2, y_minus_1, y_minus_2, w) w(1)*u_minus_1 + w(2)*(u_minus_1)^2 + w(3)*(u_minus_1)^3 + w(4)*(u_minus_1)^4+ w(5)*(u_minus_2) + w(6)*(u_minus_2)^2 +w(7)*(u_minus_2)^3 + w(8)*(u_minus_2)^4 +  w(9)*(y_minus_1)+ w(10)*(y_minus_1)^2 + w(11)*(y_minus_1)^3 + w(12)*(y_minus_1)^4 + w(13)*(y_minus_2)+ w(14)*(y_minus_2)^2 + w(15)*(y_minus_2)^3+ w(16)*(y_minus_2)^4; 

    %  dane uczące

    % bez rekurencji
    y_mod_o2_d4_arx_train = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d4_arx_train(i) = f_o2_d4(u_training(k_start-2+i), u_training(k_start-3+i), y_training(k_start-2+i),  y_training(k_start-3+i), w_o2_d4);
    end

    % z rekurencją
    y_mod_o2_d4_oe_train = zeros(length_k, 1);
    y_mod_o2_d4_oe_train(1) = y_mod_o2_d4_arx_train(1);
    y_mod_o2_d4_oe_train(2) = y_mod_o2_d4_arx_train(2);
    for i = 3:length_k
        y_mod_o2_d4_oe_train(i) =  f_o2_d4(u_training(k_start-2+i), u_training(k_start-3+i), y_mod_o2_d4_oe_train(i - 1), y_mod_o2_d4_oe_train(i - 2), w_o2_d4);
    end

    % dane weryfikujące

    % bez rekurencji
    y_mod_o2_d4_arx_valid = zeros(length_k, 1);
    for i = 1:length_k
        y_mod_o2_d4_arx_valid(i) = f_o2_d4(u_validation(k_start-2+i), u_validation(k_start-3+i), y_validation(k_start-2+i),  y_validation(k_start-3+i), w_o2_d4);
    end

    % z rekurencją
    y_mod_o2_d4_oe_valid = zeros(length_k, 1);
    y_mod_o2_d4_oe_valid(1) = y_mod_o2_d4_arx_valid(1);
    y_mod_o2_d4_oe_valid(2) = y_mod_o2_d4_arx_valid(2);

    for i = 3:length_k
        y_mod_o2_d4_oe_valid(i) = f_o2_d4(u_validation(k_start-2+i), u_validation(k_start-3+i), y_mod_o2_d4_oe_valid(i - 1), y_mod_o2_d4_oe_valid(i - 2), w_o2_d4);
    end


    plot_model_with_data_zad_2_c(k_used, y_mod_o2_d4_arx_train, y_mod_o2_d4_oe_train, y_training(k_start:end), ...
        y_mod_o2_d4_arx_valid, y_mod_o2_d4_oe_valid, y_validation(k_start:end), 2, 4)  
end



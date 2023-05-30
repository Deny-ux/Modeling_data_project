function [] = plot_model_with_data(k, y_mod_train_arx, ...
    y_mod_train_oe, y_traininig_used, y_mod_valid_arx, ...
    y_mod_valid_oe , y_validation_used)


    % liczenie błędów
    rmse_arx_train = find_MSE(y_traininig_used, y_mod_train_arx);
    rmse_oe_train = find_MSE(y_traininig_used, y_mod_train_oe);
    rmse_arx_valid = find_MSE(y_validation_used, y_mod_valid_arx);
    rmse_oe_valid = find_MSE(y_validation_used, y_mod_valid_oe);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % dane uczące
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    figure;
    % wykres bez rekurencji
    subplot(2,1,1);
    plot(k, y_traininig_used, 'b-', k, y_mod_train_arx, 'r-');
    legend('Dane uczące', 'Model', 'Location', 'best')
    xlabel('Numer próbki, k' )
    ylabel('Sygnał wyjścwiowy, y')
    title(sprintf('Dynamiczny model liniowy 1. rzędu bez rekurencji, błąd = %.7f', rmse_arx_train))
    
    % wykres z rekurencją
    subplot(2,1,2);
    plot(k, y_traininig_used, 'b-', k,  y_mod_train_oe, 'r-');
    legend('Dane uczące', 'Model', 'Location', 'best')
    xlabel('Numer próbki, k' )
    ylabel('Sygnał wyjścwiowy, y')
    title(sprintf('Dynamiczny model liniowy 1. rzędu z rekurencją, błąd = %.7f', rmse_oe_train))

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % dane weryfikujące
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    figure;
    % wykres bez rekurencji
    subplot(2,1,1);
    plot(k, y_validation_used, 'b-', k, y_mod_valid_arx, 'r-');
    legend('Dane weryfikujące', 'Model', 'Location', 'best')
    xlabel('Numer próbki, k' )
    ylabel('Sygnał wyjścwiowy, y')
    title(sprintf('Dynamiczny model liniowy 1. rzędu bez rekurencji, błąd = %.7f', rmse_arx_valid))
    
    % wykres z rekurencją
    subplot(2,1,2);
    plot(k,  y_validation_used, 'b-', k,  y_mod_valid_oe, 'r-');
    legend('Dane weryfikujące', 'Model', 'Location', 'best')
    xlabel('Numer próbki, k' )
    ylabel('Sygnał wyjścwiowy, y')
    title(sprintf('Dynamiczny model liniowy 1. rzędu z rekurencją, błąd = %.7f', rmse_oe_valid))

end
function [] = plot_static_model(fn, u_training, y_training, ...
    u_validation, y_validation, order)

    y_mod_train_n = zeros(100, 1);
    for i = 1:100
        y_mod_train_n(i) = fn(u_training(i));
    end
    rm_train = find_MSE(y_training, y_mod_train_n)

    y_mod_valid_n = zeros(100, 1);
    
    for i = 1:100
        y_mod_valid_n(i) = fn(u_validation(i));
    end
    
    rm_valid = find_MSE(y_validation, y_mod_valid_n)


    figure;
    fplot(fn, [-1, 1], 'red'); 
    hold on;
    scatter(u_training ,y_training, 'blue')
    xlabel('Sygnał wejścwiowy, u' )
    ylabel('Sygnał wyjścwiowy, y')
    % title(fprintf('Statyczny model nieliniowy, N = %d, błąd = %.4f', order, rm_train))
    if order == 1
        title(sprintf('Statyczny model liniowy, N = 1, błąd = %.4f', rm_train))
    else
        title(sprintf('Statyczny model nieliniowy, N = %d, błąd = %.4f', order, rm_train))
    end

    legend('model', 'dane uczące')
    hold off;

    figure;
    fplot(fn, [-1, 1], 'red'); 
    hold on;
    scatter(u_validation ,y_validation, 'blue')
    xlabel('Sygnał wejścwiowy, u' )
    ylabel('Sygnał wyjścwiowy, y')
    if order == 1
        title(sprintf('Statyczny model liniowy, N = 1, błąd = %.4f', rm_valid))
    else
        title(sprintf('Statyczny model nieliniowy, N = %d, błąd = %.4f', order, rm_valid))
    end
    legend('model', 'dane weryfikujące')
    hold off;
end
function [rmse] = find_MSE(real_data, app_data)
    rmse = 0;
    for i = 1:length(real_data)
        diff = (app_data(i) - real_data(i))^2;
        rmse = rmse + diff;
    end
    % diff = (app_data - real_data);
    % squaredDifference = diff.^2;
    % rmse = mean(squaredDifference);
end
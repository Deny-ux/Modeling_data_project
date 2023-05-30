function [] = plot_dynamic_model(fn, u, y, title_text, legend_text1, legend_text_2)
    fplot(fn, [-1, 1], 'red'); 
    hold on;
    scatter(u ,y, 'blue')
    xlabel('Sygnał wejścwiowy, u' )
    ylabel('Sygnał wyjścwiowy, y')
    title(title_text)
    legend(legend_text1, legend_text_2)
    hold off;
end
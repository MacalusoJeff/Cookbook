import plotnine as p9
from plotnine import ggplot, aes, geom_point, geom_line, labs

p9.options.figure_size = (12, 7)  # Increase the size of all plots

# Line plot of percentiles over time
(ggplot(results, aes(x='Date', y='Value', color='pd.Categorical(Percentile)', group='Percentile')) +
    geom_line() +
    labs(title='Percentiles over Time', x='Date', y='Value', color='Percentile') +
    p9.scale_x_date(date_breaks='1 month', date_labels='%b %Y') +
    p9.guides(color=p9.guide_legend(reverse=True)) +  # Make the legend match the order of the plot
    p9.theme(axis_text_x=p9.element_text(rotation=45, hjust=1)))

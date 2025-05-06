import pandas as pd

import src.working_utils.create_dir as cd
import src.visualization_utils.boxplots as bp
import src.visualization_utils.violinplots as vp
import src.visualization_utils.error_bars as eb
import src.visualization_utils.histograms as h
import src.visualization_utils.correlation_heatmap as ch
import src.visualization_utils.piecharts as pie

def data_visualizations(dataset: pd.DataFrame):
    dir = 'visualisations'
    cd.create_dir(dir)

    compare_column = 'Exam_Score'

    # pie charts
    pie.piecharts(dataset, dir)

    # boxplots and violinplots
    bp.boxplots(dataset, dir, compare_column)
    vp.violinplots(dataset, dir, compare_column)

    # error bars
    eb.line_charts_with_error_bars(dataset, dir, compare_column)
    eb.error_bars(dataset, dir)
    
    # histograms
    h.histograms(dataset, dir, compare_column)
    h.histograms_with_hue(dataset, dir, compare_column, 'Gender')

    # correlation heatmap
    ch.correlation_heatmap(dataset, dir)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import src.working_utils.create_dir as cd

plt.style.use('ggplot')

def line_charts_with_error_bars(dataset: pd.DataFrame, dir, compare_column):
    dir = dir + '/line_charts'
    cd.create_dir(dir)

    for column in dataset:
        if np.issubdtype(dataset[column].dtype, np.number) and column != compare_column:
            sns.relplot(
                data=dataset,
                kind='line',
                x=column,
                y=compare_column,
                errorbar="sd",
                color='hotpink'
            )

            plt.savefig('{0}/{1}_and_{2}.png'.format(dir, column, compare_column))
            plt.close()


def error_bars(dataset: pd.DataFrame, dir):
    dir = dir + '/error_bars'
    cd.create_dir(dir)

    for column in dataset:
        if np.issubdtype(dataset[column].dtype, np.number):
            f, axs = plt.subplots(2, figsize=(7, 7), sharex=True, layout="tight")
            sns.pointplot(
                    x=dataset[column],
                    errorbar='sd',
                    capsize=.3,
                    ax=axs[0],
                    color='purple'
                )
            sns.stripplot(
                    x=dataset[column],
                    jitter=.45,
                    ax=axs[1],
                    color='purple'
                )
            
            plt.savefig('{0}/{1}.png'.format(dir, column))
            plt.close()
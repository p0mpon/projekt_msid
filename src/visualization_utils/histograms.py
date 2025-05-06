import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import src.working_utils.create_dir as cd

plt.style.use('ggplot')

def histograms(dataset: pd.DataFrame, dir, compare_column):
    dir = dir + '/histograms'
    cd.create_dir(dir)

    for column in dataset:
        if np.issubdtype(dataset[column].dtype, np.number):
            sns.displot(
                    data=dataset,
                    x=column,
                    color='hotpink'
                )

            plt.savefig('{0}/{1}_and_{2}.png'.format(dir, column, compare_column))
            plt.close()


def histograms_with_hue(dataset: pd.DataFrame, dir, compare_column, hue):
    dir = dir + '/histograms_with_hue'
    cd.create_dir(dir)

    for column in dataset:
        if np.issubdtype(dataset[column].dtype, np.number) and column != hue:
            sns.displot(
                    data=dataset,
                    x=column,
                    hue=hue,
                    element='step'
                )

            plt.savefig('{0}/{1}_and_{2}.png'.format(dir, column, compare_column))
            plt.close()
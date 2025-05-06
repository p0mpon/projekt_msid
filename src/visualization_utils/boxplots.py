import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import src.working_utils.create_dir as cd

plt.style.use('ggplot')

def boxplots(dataset: pd.DataFrame, dir, compare_column):
    dir = dir + '/boxplots'
    cd.create_dir(dir)

    for column in dataset:
            if not np.issubdtype(dataset[column].dtype, np.number) and column != 'Learning_Disabilities' and column != compare_column:
                sns.catplot(
                        data=dataset,
                        x=column,
                        y=compare_column,
                        hue='Learning_Disabilities',
                        kind='box'
                    )
                
                plt.savefig('{0}/{1}_and_{2}_with_Learning_Disabilities_hue.png'.format(dir, column, compare_column))
                plt.close()
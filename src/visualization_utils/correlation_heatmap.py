import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import src.working_utils.create_dir as cd

plt.style.use('ggplot')

def correlation_heatmap(dataset: pd.DataFrame, dir):
    dir = dir + '/correlation'
    cd.create_dir(dir)

    plt.figure(figsize=(8, 6))
    num_df = dataset.select_dtypes(include=['number'])
    corr_matrix = num_df.corr()
    sns.heatmap(corr_matrix, annot=True, linewidths=0.5)
    plt.title("Correlation Heatmap\n")
    plt.tight_layout()

    plt.savefig('{0}/correlation_heatmap.png'.format(dir))
    plt.close()
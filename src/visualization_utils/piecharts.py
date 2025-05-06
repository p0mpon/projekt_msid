import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import src.working_utils.create_dir as cd

plt.style.use('ggplot')

def piecharts(dataset: pd.DataFrame, dir):
    dir = dir + '/piecharts'
    cd.create_dir(dir)

    for column in dataset:
        if not np.issubdtype(dataset[column].dtype, np.number):
            counts = dataset[column].value_counts()

            plt.figure(figsize=(6,6))
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['gold', 'lightcoral', 'lightskyblue'])
            plt.title(column)
                
            plt.savefig('{0}/{1}.png'.format(dir, column))
            plt.close()
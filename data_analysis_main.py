import src.working_utils.import_data as im
import src.stat_utils.data_statistics as ds
import src.visualization_utils.data_visualizations as dv

def main():
    dataset = im.import_data()

    # write statistics to .csv files in /statistics
    ds.data_statistics(dataset)

    # save visualizations to .png files in /visualizations
    dv.data_visualizations(dataset)


if __name__ == '__main__':
    main()
import pandas as pd
import os


def csv_to_modelica_input(source_file, target_file):
    """
    Converts csv file to txt file suitable for loading into Modelica

    :param source_file: path to csv file to be converted
    :param target_file: path to txt file to be made
    """

    df = pd.read_csv(source_file, sep=';', index_col=0)
    shape = (len(df.index), len(df.columns)+1)
    df.to_csv('tempfile.txt', header=None, sep=' ')
    header = '#1\ndouble data' + str(shape) + '\n'

    with open('tempfile.txt', 'r') as f:
        with open(target_file, 'w') as f2:
            f2.write(header)
            f2.write(f.read())
    os.remove('tempfile.txt')


if __name__ == '__main__':

    wp_dir = os.path.abspath('../..')  # Work package directory
    results_dir = os.path.join(wp_dir, 'Results', 'SimpleDistrict')
    data_dir = os.path.join(wp_dir, 'Modelica', 'IBPSAdestest', 'Resources', 'Data', 'DestestHeatDemand')
    csv_to_modelica_input(os.path.join(results_dir, 'SimpleDistrict_district.csv'),
                          os.path.join(data_dir, 'heat_profiles.txt'))

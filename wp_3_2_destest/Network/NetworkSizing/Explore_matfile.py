import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
from modelicares import SimRes


def main():
    st.title("Plot Jonas Destest")

    res_dir = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/mat_files_from_models"

    res_path = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/model" \
               "/Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50" \
               "/Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50_inputs.mat"

    dir_this = os.path.abspath(os.path.dirname(__file__))  # saves the path where this file is executed to a string
    dir_src = os.path.abspath(os.path.dirname(dir_this))  # saves one higher path to a string
    # dir_top = os.path.abspath(os.path.dirname(dir_src))
    dir_workspace = os.path.abspath(os.path.join(dir_src, "workspace"))  # saves a new workspace path to a string
    if not os.path.exists(dir_workspace):
        os.makedirs(dir_workspace)
    dir_output = os.path.abspath(os.path.join(dir_workspace, "plots"))  # saves a plot path to a string
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    pipes, substations_vars, supply_vars = read_variable_names_filtered(res_path=res_path)

    sub_var_lst = [i for i in supply_vars if "m_flow" in i]  # instead of i.endswith("str")
    print(sub_var_lst)

    res_dict = variable_names_filtered_to_csv(res_dir=res_dir, res_path=res_path, var_lst=sub_var_lst,
                                              savename_append="_pumping_power_modelicares.csv")
    # res_dict   # streamlit magic

    for key in res_dict.keys():
        plot_from_df(res_df=res_dict[key], key=key, dir_output=dir_output,
                     var_to_plot="networkModel.supplyDestest_Supply.senMasFlo.m_flow")


@st.cache(persist=True)
def import_simulation_results_modelicares_to_df(res_path, var_lst):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. The variables given in the var_lst list
    are then converted to a pandas dataframe.   (... further explanation ...)
    The dataframe is then returned.

    :param res_path: str
        Path to the .mat result file
    :param var_lst: list
        List of variable names
    :return: res_df: pandas dataframe
        A pandas dataframe containing the chosen results.
    """
    print("Converting the Matfile to a SimRes Instance ...")
    sim = SimRes(res_path)
    print("Converting the SimRes Instance to a Pandas Dataframe ...")
    res_df = sim.to_pandas(var_lst, with_unit=False)

    res_df = res_df.groupby(res_df.index).first()
    res_df = res_df[res_df.index.isin(range(0, 31536000, 900))]

    res_df.index = res_df.index.astype(int)
    res_df.index = pd.to_datetime(res_df.index, unit="s", origin="2019")

    return res_df


@st.cache(persist=True)
def variable_names_filtered_to_csv(res_dir, res_path, var_lst, savename_append, save_to_csv=False):
    """
    Imports results with 'import_simulation_results_modelicares_to_df' function. Then converts the dataframe to a csv.
    Adds the dataframe with the corresponding Result path to a dictionary. (This might be useful if multiple result
    fields are analysed at once?)

    :param res_dir: str
        Path to the results directory
    :param res_path: str
        Path to the result file
    :param var_lst: list
        list of variables to import from result file
    :param savename_append: str
        String to append to the csv title
    :param save_to_csv: bool
        boolean to decide if the results should be saved to a csv file
    :return res_dict: dictionary
        dictionary for looping multiple result files? or necessary for streamlit magic stuff?
    """

    res_all_lst = [os.path.join(res_dir, file) for file in os.listdir(res_dir) if
                   (os.path.isfile(os.path.join(res_dir, file)) and file.endswith(".mat"))]
    res_dict = {}

    for file in res_all_lst:
        res_df = import_simulation_results_modelicares_to_df(res_path=file, var_lst=var_lst)
        if save_to_csv:
            print("Converting the Dataframe to a CSV ... This could take some minutes!")
            savename_csv = file.split("/")[-1][:-4] + savename_append
            res_df.to_csv(savename_csv)  # prints an overview over the csv and saves it to the current folder(?)
        res_dict[file] = res_df

    return res_dict


def read_variable_names_filtered(res_path):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered trajectories are saved in a list. A trajectory is considered a variable which has more than two values,
    or the values are not equal. The 'get_trajectories' function exists only in the EBC branch of the ModelicaRes
    package. From the 'all_trajectory_names', three sublists are created, for pipe, demand and supply trajectories.
    These sublists are returned.

    :param res_path: str
        Path to the result file
    :return pipe_names, substation_names, supply_names : lists
        Lists that contain the names of the associated variables/trajectories
    """
    print("Converting the .mat File to a SimRes instance ...")
    sim = SimRes(fname=res_path)
    all_trajectory_names = sim.get_trajectories()  # all_trajectory_names is a list of variables that are trajectories.

    # add all pipe variables from the all_trajectory_names into a new list
    pipe_names = [i for i in all_trajectory_names if "Pipe" in i]
    print("There are " + str(len(pipe_names)) + " Pipe variables in the results file")

    # add all demand variables from the all_trajectory_names into a new list
    substation_names = [i for i in all_trajectory_names if "demand" in i]
    print("There are " + str(len(substation_names)) + " Substation variables in the results file")

    # add all supply variables from the all_trajectory_names into a new list
    supply_names = [i for i in all_trajectory_names if "networkModel.supply" in i]
    print("There are " + str(len(supply_names)) + " Supply variables in the results file")

    return pipe_names, substation_names, supply_names


def plot_from_df(res_df, key, dir_output, var_to_plot, save_figs=False):
    """

    :param res_df:
    :param key:
    :param dir_output:
    :param var_to_plot:
    :param save_figs:
    :return:
    """

    name = key.split("/")[-1][:-4]
    fig1, ax1 = plt.subplots()
    fig1.suptitle(name, fontsize=5)

    ax1.plot(
        res_df[var_to_plot].resample("H").mean(),  # 'D' prints daily means
        label="Average Mass Flow Supply: %s kg/s" % (
            round(res_df[var_to_plot].mean(), 2)))

    ax1.set_ylabel("Mass Flow in kg/s")
    ax1.set_xlabel("Time in Date")

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0.)

    fig1.autofmt_xdate(rotation=45, ha="center")

    if save_figs:
        fig1.savefig(os.path.join(dir_output, name + "_pumping_power.pdf"))
        fig1.savefig(os.path.join(dir_output, name + "_pumping_power.png"), dpi=600)

    st.pyplot(fig1)

    return res_df


if __name__ == '__main__':
    main()

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

    # folder management
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

    sub_var_lst = make_sublist(input_var_lst=supply_vars, sub_var_signal=".")

    res_dict = variable_names_filtered_to_csv(res_dir=res_dir, res_path=res_path, var_lst=sub_var_lst,
                                              savename_append="_pumping_power_modelicares.csv")
    # res_dict   # streamlit magic

    for key in res_dict.keys():
        plot_from_df(res_df=res_dict[key], key=key, dir_output=dir_output,
                     var_to_plot="networkModel.supplyDestest_Supply.senMasFlo.m_flow")


def make_sublist(input_var_lst, sub_var_signal):
    """
    Makes a sublist of a list, depending on the signal word given.
    :param input_var_lst:
    :param sub_var_signal:
    :return:
    """
    sub_var_lst = [i for i in input_var_lst if sub_var_signal in i]  # instead of i.endswith("str")
    print("There're " + str(len(input_var_lst)) + " Variables inside the main list, that contain "
          + sub_var_signal, ":", str(sub_var_lst))

    return sub_var_lst


# @st.cache(persist=True)
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


# @st.cache(persist=True)
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

    print("Importing " + str(len(res_all_lst)) + " .mat files to Dataframes")
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
    pipe_names = [i for i in all_trajectory_names if "pipe" in i]
    print("There are " + str(len(pipe_names)) + " Pipe variables in the results file: " + str(pipe_names))

    # add all demand variables from the all_trajectory_names into a new list
    substation_names = [i for i in all_trajectory_names if "demand" in i]
    print("There are " + str(len(substation_names)) +
          " Substation variables in the results file: " + str(substation_names))

    # add all supply variables from the all_trajectory_names into a new list
    supply_names = [i for i in all_trajectory_names if "networkModel.supply" in i]
    print("There are " + str(len(supply_names)) + " Supply variables in the results file: " + str(supply_names))

    return pipe_names, substation_names, supply_names


def plot_from_df(res_df, key, dir_output, var_to_plot, save_figs=False):
    """
    Creates a Matplotlib figure and plots the variable in 'var_to_plot'. Saves the figure in 'dir_output'

    :param res_df:
    :param key:
    :param dir_output:
    :param var_to_plot:
    :param save_figs:
    :return:
    """

    name = key.split("/")[-1][:-4]

    # Figure 1
    fig1, ax1 = plt.subplots()
    fig1.suptitle(name, fontsize=5)

    ax1.plot(res_df[var_to_plot].resample("D").mean(),  # 'D' prints daily means
             label="Average Mass Flow Supply: %s kg/s" % (round(res_df[var_to_plot].mean(), 2)))
    ax1.set_ylabel("Mass Flow in kg/s")
    ax1.set_xlabel("Time in Date")
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0.)
    fig1.autofmt_xdate(rotation=45, ha="center")

    if save_figs:
        fig1.savefig(os.path.join(dir_output, name + "_mass_flow.pdf"))
        fig1.savefig(os.path.join(dir_output, name + "_mass_flow.png"), dpi=600)

    # st.pyplot(fig1)
    print(fig1)

    # Figure 2, Ax2
    fig2, ax2 = plt.subplots()
    fig2.suptitle(name, fontsize=5)

    line1 = ax2.plot(
        res_df["networkModel.supplyDestest_Supply.senMasFlo.port_b.p"].resample("D").mean() / 100000 -
        res_df["networkModel.supplyDestest_Supply.senMasFlo.port_a.p"].resample("D").mean() / 100000,
        label=r"Pressure at supply: $\varnothing$ %s Bar" % (
            round(res_df["networkModel.supplyDestest_Supply.senMasFlo.port_a.p"].sum() / 100000 / 8760, 2)),
        color="m")
    ax2.set_ylabel(r"Pressure Head in bar")
    # ax2.set_ylim(4.5, 6.5)
    ax2.set_xlabel(r"Time in Date")

    # Figure 2, Ax3
    ax3 = ax2.twinx()
    line2 = ax3.plot(
        res_df["networkModel.supplyDestest_Supply.port_b.m_flow"].resample("D").min() * -1,
        label="Massflow rate" % (round(res_df["networkModel.supplyDestest_Supply.port_b.p"].sum() / 100000 / 8760, 2)),
        color="g")
    ax3.set_ylabel(r"Massflow in kg/s")
    ax3.set_ylim(2, 18)

    lines = line1 + line2
    labs = [line.get_label() for line in lines]

    ax2.legend(lines, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0., ncol=2)

    fig2.autofmt_xdate(rotation=45, ha="center")

    if save_figs:
        fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.pdf"))
        fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.png"), dpi=600)

    st.pyplot(fig2)
    print(fig2)

    return fig1, fig2


if __name__ == '__main__':
    main()

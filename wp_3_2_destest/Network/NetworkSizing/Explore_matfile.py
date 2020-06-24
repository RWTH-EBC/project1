import pandas as pd
import os
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
from modelicares import SimRes


def main():
    st.title("Plot Jonas Destest")

    # res_dir = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/mat_files_from_models"
    res_dir = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/model"

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

    all_vars_lst = read_trajectory_names(res_path=res_path)

    sublist_m_flow = make_sublist(all_vars_lst, "senMasFlo.m_flow", "", "R")
    sublist_m_flow_return = make_sublist(all_vars_lst, "senMasFlo.m_flow", "R", "")
    all_sup_ret_temps = make_sublist(all_vars_lst, "senT", ".T", "")
    supply_vars = make_sublist(all_vars_lst, "networkModel.supply")

    res_dict = variable_names_filtered_to_csv(res_dir=res_dir, var_lst=all_sup_ret_temps,
                                              savename_append="all_sup_ret_temps.csv")

    # streamlit magic
    sublist_m_flow
    sublist_m_flow_return
    all_sup_ret_temps
    supply_vars

    for key in res_dict.keys():
        plot_from_df_looped(res_df=res_dict[key], key=key, dir_output=dir_output, var_lst=all_sup_ret_temps,
                            ylabel="Supply and Return Temps", savename_append="_massflow_pressure")


def plot_from_df_looped(res_df, key, dir_output, var_lst, ylabel, savename_append, save_figs=True):
    """
    Creates a Matplotlib figure and plots the variable in 'var_to_plot'. Saves the figure in 'dir_output'

    :param ylabel: string
    :param var_lst: list
    :param res_df: pandas dataframe:    result file
    :param key: String:                 path to result .mat file
    :param dir_output: String:          path to output directory
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """

    name = key.split("/")[-1][:-4]
    fig_lst = []  # for return statement

    fig, ax = plt.subplots()
    fig.suptitle(name, fontsize=5)
    lines = []

    for var in var_lst:
        line = ax.plot(res_df[var].resample("D").mean(), linewidth=0.7,
                       label=''.join(c for c in str(var) if c.isdigit()) if "Simple" in str(var) else str(var)[27:])
        lines += line

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time in Date")
    labs = [line.get_label() for line in lines]
    ax.legend(lines, labs, loc="best", borderaxespad=0., ncol=2, fontsize=5)
    fig.autofmt_xdate(rotation=45, ha="center")

    st.pyplot(fig)
    fig_lst.append(fig)

    if save_figs:
        fig.savefig(os.path.join(dir_output, name, "_", savename_append + ".pdf"))
        fig.savefig(os.path.join(dir_output, name, "_", savename_append + ".png"), dpi=600)

    return fig_lst


def plot_from_df(res_df, key, dir_output, var_to_plot, var_lst, save_figs=True, plt_fig1=True, plt_fig2=True,
                 plt_fig3=True):
    """
    Creates a Matplotlib figure and plots the variable in 'var_to_plot'. Saves the figure in 'dir_output'

    :param var_lst:
    :param res_df: pandas dataframe:    result file
    :param key: String:                 path to result .mat file
    :param dir_output: String:          path to output directory
    :param var_to_plot: String:         variable name from the results pandas dataframe
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :param plt_fig1: boolean:           decision to create and plot figure 1
    :param plt_fig2: boolean:           decision to create and plot figure 2
    :param plt_fig3: boolean:           decision to create and plot figure 3
    :return: fig_lst: list:             list that contains all created figures
    """

    name = key.split("/")[-1][:-4]
    fig_lst = []  # for return statement

    # Figure 1
    if plt_fig1:
        fig1, ax1 = plt.subplots()
        fig1.suptitle(name, fontsize=10)

        ax1.plot(res_df[var_to_plot].resample("D").mean(),  # 'D' prints daily means
                 label="Average Mass Flow Supply: %s kg/s" % (round(res_df[var_to_plot].mean(), 2)))
        ax1.set_ylabel("Mass Flow in kg/s")
        ax1.set_xlabel("Time in Date")
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0.)
        fig1.autofmt_xdate(rotation=45, ha="center")

        if save_figs:
            fig1.savefig(os.path.join(dir_output, name + "_mass_flow.pdf"))
            fig1.savefig(os.path.join(dir_output, name + "_mass_flow.png"), dpi=600)

        st.pyplot(fig1)
        fig_lst.append(fig1)

    # Figure 2, Ax2
    if plt_fig2:
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
            label="Massflow rate" % (
                round(res_df["networkModel.supplyDestest_Supply.port_b.p"].sum() / 100000 / 8760, 2)),
            color="g")
        ax3.set_ylabel(r"Massflow in kg/s")
        # ax3.set_ylim(2, 18)

        lines = line1 + line2
        labs = [line.get_label() for line in lines]

        ax2.legend(lines, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0., ncol=2)

        fig2.autofmt_xdate(rotation=45, ha="center")

        if save_figs:
            fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.pdf"))
            fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.png"), dpi=600)

        st.pyplot(fig2)
        fig_lst.append(fig2)

    return fig_lst


def make_sublist(input_var_lst, sig1="", sig2="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal word given.

    :param input_var_lst: list
        input list
    :param sig1: String
        String that should be included in the sublists results
        Examples: pipe, demand, networkModel.supply
    :param sig2: String
        String that should be included in the sublists results
        Examples: .vol.T,
    :param anti_sig1: String
        String that shouldn't be included in th sublists results
        Examples: R,
    :return sub_var_lst1: list
        sublist

    """
    sublist = [i for i in input_var_lst if sig1 in i]
    if sig2:
        sublist = [i for i in sublist if sig2 in i]  # instead of i.endswith("str")
    if anti_sig1:
        sublist = [i for i in sublist if anti_sig1 not in i]
    # print("There're " + str(len(sublist)) + " Variables inside the main list, that contain "
    #       + sig1, sig2, "and don't contain ", anti_sig1, ":", str(sublist))

    return sublist


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
def variable_names_filtered_to_csv(res_dir, var_lst, savename_append, save_to_csv=False):
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
    res_dict = {}
    res_all_lst = []

    for folders, sub_folders, files in os.walk(res_dir):
        for file in files:
            res_all_lst.append(os.path.join(folders, file)) if file.endswith(".mat") else None

    print("Importing " + str(len(res_all_lst)) + " .mat files to Dataframes")
    for mat_file in res_all_lst:
        res_df = import_simulation_results_modelicares_to_df(res_path=mat_file, var_lst=var_lst)
        if save_to_csv:
            print("Converting the Dataframe to a CSV ... This could take some minutes!")
            savename_csv = mat_file.split("/")[-1][:-4] + savename_append
            res_df.to_csv(savename_csv)  # prints an overview over the csv and saves it to the current folder(?)
        res_dict[mat_file] = res_df

    return res_dict


def read_trajectory_names(res_path):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered trajectories are saved in a list. A trajectory is considered a variable which has more than two values,
    or the values are not equal. The 'get_trajectories' function exists only in the EBC branch of the ModelicaRes
    package.

    :param res_path: str
        Path to the result file
    :return all_trajectories_lst : list
        List that contain the names of all variables/trajectories
    """
    print("Converting the .mat File to a SimRes instance ...")
    sim = SimRes(fname=res_path)
    all_trajectories_lst = sim.get_trajectories()  # all_trajectory_names is a list of variables that are trajectories.
    print(
        "There are " + str(len(all_trajectories_lst)) + " variables in the results file: " + str(all_trajectories_lst))

    return all_trajectories_lst


if __name__ == '__main__':
    main()

import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
from modelicares import SimRes
import fnmatch
import platform


def main():
    st.title("Plot Jonas Destest")

    # paths
    if platform.system() == 'Darwin':
        dir_models = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/models"
        dir_output = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/plots"
        master_csv = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/overview_files/master.csv"
    elif platform.system() == 'Windows':
        dir_models = "/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/models"
        dir_output = "/mma-jgr/sciebo-folder/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/plots"
        master_csv = "/mma-jgr/sciebo-folder/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data/overview_files/master.csv"
    else:
        raise Exception("Unknown operating system")

    # saving all variables names to a list
    all_vars_lst = read_trajectory_names_folder(res_dir=dir_models)

    # making sublists
    sublist_m_flow = make_sublist(all_vars_lst, "senMasFlo.m_flow", "", "R")
    sublist_m_flow_return = make_sublist(all_vars_lst, "senMasFlo.m_flow", "R", "")
    all_supply_return_temps = make_sublist(all_vars_lst, "senT", ".T", "")
    supply_vars = make_sublist(all_vars_lst, "networkModel.supply")
    power_demand_heatpump = make_sublist(all_vars_lst, 'heaPum.P')

    # converting the mat files to dataframes
    res_dict = mat_files_filtered_to_dict(res_dir=dir_models, var_lst=all_vars_lst,
                                          savename_append="all_sup_ret_temps.csv")

    # streamlit magic, show those sublists for easy inspection
    # sublist_m_flow
    # sublist_m_flow_return
    # all_supply_return_temps
    # supply_vars

    # plot the data of the sublist variables
    plot_from_df_looped(res_dict, dir_output, plot_style=2, var_lst=all_supply_return_temps, master_csv=master_csv,
                        y_label="Supply and Return Temps")
    plot_from_df_looped(res_dict, dir_output, plot_style=2, var_lst=sublist_m_flow, master_csv=master_csv,
                        y_label="Mass Flows Supply Pipes")
    plot_from_df_looped(res_dict, dir_output, power_demand_heatpump, y_label='Power HP', plot_style=2,
                        master_csv=master_csv)


def read_trajectory_names_folder(res_dir, print_all_trajectories=False):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered trajectories are saved in a list. A trajectory is considered a variable which has more than two values,
    or the values are not equal. The 'get_trajectories' function exists only in the EBC branch of the ModelicaRes
    package.
    :param print_all_trajectories:
    :param res_dir:
    :return all_trajectories_lst:   list:   names of all variables that are trajectories
        """
    res_all_lst = find("*.mat", res_dir)
    res_path = res_all_lst[0]

    print("Converting the .mat File to a SimRes instance to get the variable names ...")
    sim = SimRes(fname=res_path)
    all_trajectories_lst = sim.get_trajectories()  # all_trajectory_names is a list of variables that are trajectories.

    if not print_all_trajectories:
        print("There are " + str(len(all_trajectories_lst)) + " variables in the results file.")
    else:
        print("There are " + str(len(all_trajectories_lst)) + " variables in the results file:"
              + str(all_trajectories_lst))

    return all_trajectories_lst


def find(pattern, path):
    """
    Finds files that match a pattern and return a list of all file paths
    :param pattern:
    :param path:
    :return:
    """
    results = []

    if not os.path.isdir(path):
        error_message = "result directory {} doesn't exist! Please update path.".format(path)
        raise Exception(error_message)

    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                results.append(os.path.join(root, name))

    if not results:
        error_message = "No File found that contains '{pattern}' in result directory {path}"\
            .format(pattern=pattern, path=path)
        raise Exception(error_message)

    return results


def make_sublist(input_var_lst, sig1="", sig2="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal words given.
    This is useful for plotting a specific subset of variables.

    :param input_var_lst: list      input list
    :param sig1: String             String that should be included in the sublists results
                                    Examples: pipe, demand, networkModel.supply
    :param sig2: String             String that should be included in the sublists results. Examples: .vol.T,
    :param anti_sig1: String        String that shouldn't be included in th sublists results. Examples: R,
    :return sub_var_lst1: list      sublist
    """
    sublist = [i for i in input_var_lst if sig1 in i]
    if sig2:
        sublist = [i for i in sublist if sig2 in i]  # instead of i.endswith("str")
    if anti_sig1:
        sublist = [i for i in sublist if anti_sig1 not in i]
    # print("There're " + str(len(sublist)) + " Variables inside the main list, that contain "
    #       + sig1, sig2, "and don't contain ", anti_sig1, ":", str(sublist))

    return sublist


@st.cache(persist=True)
def matfile_to_df(res_path, var_lst):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. The variables given in the var_lst list
    are then converted to a pandas dataframe.   (... further explanation ...) The dataframe is then returned.
    :param res_path:    str:        Path to the .mat result file
    :param var_lst:     list:       variable names
    :return: res_df:    pandas df:  results
    """
    if not os.path.isfile(res_path):
        error_message = "result directory {} doesn't exist! Please update path.".format(res_path)
        raise Exception(error_message)

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
def mat_files_filtered_to_dict(res_dir, var_lst, savename_append, save_to_csv=False):
    """
    Imports results with 'matfile_to_df' function. Then converts the dataframe to a csv.
    Adds the dataframe with the corresponding Result path to a dictionary. (This might be useful if multiple result
    fields are analysed at once?)
    :param res_dir: str:            Path to the results directory
    :param var_lst: list:           variables to import from result file
    :param savename_append: str:    name to append to the csv title
    :param save_to_csv: bool:       decide if the results should be saved to a csv file
    :return res_dict: dictionary:   for looping multiple result files, necessary for streamlit magic stuff
    """
    res_dict = {}
    mat_files = find("*.mat", res_dir)

    print("Importing {x} .mat files to Dataframes".format(x=str(len(mat_files))))
    for mat_file in mat_files:
        res_df = matfile_to_df(res_path=mat_file, var_lst=var_lst)
        if save_to_csv:
            print("Converting the Dataframe to a CSV ... This could take some minutes!")
            savename_csv = mat_file.split("/")[-1][:-4] + savename_append
            res_df.to_csv(savename_csv)  # prints an overview over the csv and saves it to the current folder(?)
        sim_name = mat_file.split("/")[-1][:-4]
        res_dict[sim_name] = res_df
    return res_dict


def plot_from_df_looped(res_dict, dir_output, var_lst, y_label, plot_style, master_csv, savename_append='',
                        save_figs=False):
    """

    Creates a Matplotlib figure and plots the variables in 'var_lst'. Saves the figure in 'dir_output'

    :param master_csv:
    :param plot_style:                  plot style 1: one figure plots all variables of a single simulation
                                        plot style 2: one figure plots a single variable of all simulations
    :param res_dict: dictionary         dictionary that holds the Simulation names as keys and the corresponding
                                        variable data as a dataframe as values
    :param y_label: string
    :param var_lst: list
    :param dir_output: String:          path to output directory
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """
    fig_lst = []  # for return statement

    if plot_style == 1:     # Single Simulation
        for sim_name in res_dict.keys():

            fig, ax = plt.subplots()
            fig.suptitle(sim_name, fontsize=5)
            lines = []

            for var in var_lst:
                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7,
                               label=''.join(c for c in str(var) if c.isdigit()) if "Simple" in str(var) else str(var)[
                                                                                                              27:])
                lines += line

            ax.set_ylabel(y_label)
            ax.set_xlabel("Time in Date")
            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0., ncol=2, fontsize=5)
            fig.autofmt_xdate(rotation=45, ha="center")

            st.pyplot(fig)
            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".pdf"))
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".png"), dpi=600)

    elif plot_style == 2:   # Single Variable
        for var in var_lst:

            fig, ax = plt.subplots()
            fig.suptitle(var, fontsize=7)
            lines = []

            # take master_csv and reduce it down to the inspected Simulations and the variable that changes (and is
            # therefore part of a simulation study). This is useful for  naming  the plotted lines.
            master_df = pd.read_csv(master_csv, index_col=0)
            master_indexes = list(master_df.index.values)
            sim_names = res_dict.keys()
            sim_names_to_drop = [x for x in master_indexes if x not in sim_names]
            master_df = master_df.drop(index=sim_names_to_drop)
            master_df = master_df[[i for i in master_df if len(set(master_df[i])) > 1]]
            master_df = master_df.dropna(axis=1, how="any")
            var0 = master_df.columns[0]

            for sim_name in res_dict.keys():
                val0 = master_df._get_value(sim_name, var0)
                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7,
                               label=str(var0) + ' = ' + str(val0))
                lines += line

            # for sim_name in

            ax.set_ylabel(y_label)
            ax.set_xlabel("Time in Date")
            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0., ncol=2, fontsize=5)
            fig.autofmt_xdate(rotation=45, ha="center")

            st.pyplot(fig)
            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".pdf"))
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".png"), dpi=600)

    return fig_lst


# unused functions
def read_trajectory_names_file(res_path):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered trajectories are saved in a list. A trajectory is considered a variable which has more than two values,
    or the values are not equal. The 'get_trajectories' function exists only in the EBC branch of the ModelicaRes
    package.
    :param res_path:                str:    Path to the result file
    :return all_trajectories_lst:   list:   names of all variables that are trajectories
    """

    print("Converting the .mat File to a SimRes instance ...")
    sim = SimRes(fname=res_path)
    all_trajectories_lst = sim.get_trajectories()  # all_trajectory_names is a list of variables that are trajectories.
    print(
        "There are " + str(len(all_trajectories_lst)) + " variables in the results file.")
    # "+ str(all_trajectories_lst)) if you wish to print out all variable names

    return all_trajectories_lst


def plot_from_df(res_df, key, dir_output, var_to_plot, save_figs=True, plt_fig1=True, plt_fig2=True):
    """
    Creates a Matplotlib figure and plots the variable in 'var_to_plot'. Saves the figure in 'dir_output'

    :param res_df: pandas dataframe:    result file
    :param key: String:                 path to result .mat file
    :param dir_output: String:          path to output directory
    :param var_to_plot: String:         variable name from the results pandas dataframe
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :param plt_fig1: boolean:           decision to create and plot figure 1
    :param plt_fig2: boolean:           decision to create and plot figure 2
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
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)
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
            if not os.path.exists(dir_output):
                os.makedirs(dir_output)
            fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.pdf"))
            fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.png"), dpi=600)

        st.pyplot(fig2)
        fig_lst.append(fig2)

    return fig_lst


if __name__ == '__main__':
    main()

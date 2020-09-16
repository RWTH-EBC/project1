import pandas as pd
import os
import matplotlib.pyplot as plt
import streamlit as st
from modelicares import SimRes
import fnmatch
import platform
from PIL import Image


def main():
    # paths
    if platform.system() == 'Darwin':
        dir_sciebo = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    elif platform.system() == 'Windows':
        dir_sciebo = "D:/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    else:
        raise Exception("Unknown operating system")
    dir_models = dir_sciebo + "/models"
    dir_output = dir_sciebo + "/plots"
    master_csv = dir_sciebo + "/overview_files/master.csv"
    study_csv = dir_sciebo + "/models/study.csv"

    # saving all variables names to a list
    # only makes sense if all the simulations have the same variables!
    # this is not the case when looking into different supply/demand/pipe models!
    all_vars_lst = read_trajectory_names_folder(res_dir=dir_models, only_from_first_sim=True)

    # making sublists
    all_sublists = {}
    add_to_sublists(all_sublists, 'Mass Flow Supply Pipes', all_vars_lst, "senMasFlo.m_flow", "pipe", "R")
    add_to_sublists(all_sublists, 'Mass Flow Return Pipes', all_vars_lst, "senMasFlo.m_flow", "R", "")
    add_to_sublists(all_sublists, 'Supply and Return Temperatures', all_vars_lst, "senT", ".T", "")
    add_to_sublists(all_sublists, 'Supply Variables', all_vars_lst, "networkModel.supply")
    add_to_sublists(all_sublists, 'Electrical Power Demand Heatpump', all_vars_lst, 'heaPum.P')
    add_to_sublists(all_sublists, 'Return Temp Substations', all_vars_lst, 'del1.T')

    # converting the mat files to dataframes
    res_dict = mat_files_filtered_to_dict(res_dir=dir_models, var_lst=all_vars_lst)

    # streamlit dashboard, print map of destest
    st.sidebar.title("Data Selection for Destest")
    destest_map = Image.open('/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing'
                             '/uesgraph_destest_16_selfsized_jonas.png')
    st.image(destest_map, caption='Destest Network Layout', use_column_width=True)

    # choose a sublist of variables from all the sublists inside the all_sublists dictionary
    selected_sublist_description = st.sidebar.radio(label="Choose which sublist of variables to plot",
                                                    options=list(all_sublists.keys()))

    # choose a plot style
    selected_plot_style = st.sidebar.radio(label="Choose which plotstyle tu use",
                                           options=['Single Simulation', 'Single Variable'])

    # choose a Variable inside the selected sublist of variables
    selected_variable_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single Variable per Plot, choose which variable to plot from the chosen sublist",
        options=all_sublists[selected_sublist_description])

    # choose a Simulation from the res_dict
    selected_simulation_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single simulation per Plot, choose which simulation to plot",
        options=list(res_dict.keys()),
        default=list(res_dict.keys()))

    # loop through all sublists, find the selected one and print it
    for sublist_description in list(all_sublists.keys()):
        if sublist_description == selected_sublist_description:
            st.write("The selected sublist contains the following variables:")
            st.write(all_sublists[sublist_description])

    # One Plot displays one Single Variable, one line is one simulation -> Dymola can't do this
    if selected_plot_style == 'Single Variable':
        vars_to_plot = []
        for var in all_sublists[selected_sublist_description]:
            if var == selected_variable_from_sublist:
                vars_to_plot.append(selected_variable_from_sublist)
        fig_lst = plot_from_df_looped(res_dict, dir_output, plot_style='Single Variable', var_lst=vars_to_plot,
                                      study_csv=study_csv, y_label=selected_sublist_description)
        for fig in fig_lst:
            st.pyplot(fig)

    # One Plot displays one Single Simulation, one line is one variable -> Dymola can do this, too
    if selected_plot_style == 'Single Simulation':
        plot_sim_dict = {}
        for sim in list(res_dict.keys()):
            if sim in selected_simulation_from_sublist:
                plot_sim_dict.update({sim: res_dict[sim]})
        fig_lst = plot_from_df_looped(plot_sim_dict, dir_output, plot_style="Single Simulation",
                                      var_lst=all_sublists[selected_sublist_description], study_csv=study_csv,
                                      y_label=selected_sublist_description)
        for fig in fig_lst:
            st.pyplot(fig)


def read_trajectory_names_folder(res_dir, only_from_first_sim):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered trajectories are saved in a list. A trajectory is considered a variable which has more than two values,
    or the values are not equal. The 'get_trajectories' function exists only in the EBC branch of the ModelicaRes
    package.
    :param only_from_first_sim: decide if you want to take the trajectory names only from the first simulation of a
                                parameter study or of all simulations. It makes sense to take the trajectories of all
                                simulations, if the variable names change. For example, when using different substation
                                or supply models within the same simulation study.
    :param res_dir:
    :return all_trajectories_lst:   list:   names of all variables that are trajectories
        """
    res_all_lst = find("*.mat", res_dir)

    if only_from_first_sim:
        res_path = res_all_lst[0]
        print("Converting one .mat File to a SimRes instance to get the variable names ...")
        sim = SimRes(fname=res_path)
        all_trajectories_lst = sim.get_trajectories()  # all_trajectory_names is list of variables that are trajectories
        print("There are " + str(len(all_trajectories_lst)) + " variables in the first results file.")
    else:
        all_trajectories_lst = []
        print("Converting the .mat Files to SimRes instances to get all variable names ...")
        for res_path in res_all_lst:
            sim = SimRes(fname=res_path)
            sim_trajectories_lst = sim.get_trajectories()
            all_trajectories_lst.extend(sim_trajectories_lst)
        all_trajectories_lst = list(set(all_trajectories_lst))  # drops duplicates

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
        error_message = "No File found that contains '{pattern}' in result directory {path}" \
            .format(pattern=pattern, path=path)
        raise Exception(error_message)

    return results


def make_sublist(description, input_var_lst, sig1="", sig2="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal words given. This is useful for plotting a specific
    subset of variables. Returns the list together with its description as a dictionary.

    :param description: String      description of the variables inside the output sublist
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

    return {description: sublist}


def add_to_sublists(master_sublists_dict, description, input_var_lst, sig1="", sig2="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal words given. This is useful for plotting a specific
    subset of variables. Returns the list together with its description as a key:value pair. Adds this pair to the
    master_sublists_dict that contains all those key:value pairs.

    :param master_sublists_dict:    dictionary that stores all the sublists as values and their description as keys
    :param description: String      description of the variables inside the output sublist
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

    master_sublists_dict.update({description: sublist})

    return {description: sublist}


# @st.cache(persist=True)
def matfile_to_df(res_path, var_lst, output_interval="hours"):
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

    if output_interval == "seconds":
        res_df = res_df[res_df.index.isin(range(0, 31536000, 900))]
        res_df.index = res_df.index.astype(int)
        res_df.index = pd.to_datetime(res_df.index, unit="s", origin="2019")
    elif output_interval == "hours":
        res_df = res_df[res_df.index.isin(range(0, 8760, 1))]
        res_df.index = res_df.index.astype(int)
        # res_df.index = pd.to_datetime(res_df.index, unit="h", origin="2019") #-> to_datetime has no Unit "Hour"
        res_df.index = pd.Timestamp('2019-01-01') + pd.to_timedelta(res_df.index, unit='H')
    return res_df


# @st.cache(persist=True)
def mat_files_filtered_to_dict(res_dir, var_lst, savename_append="", save_to_csv=False):
    """
    Imports results with 'matfile_to_df' function. Optionally converts the dataframe to a csv.
    Adds the dataframe with the corresponding Result path to a dictionary. (This might be useful if multiple result
    fields are analysed at once)
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
            savename_csv = mat_file.split("/")[-1][:-4], savename_append
            res_df.to_csv(savename_csv)  # prints an overview over the csv and saves it to the current folder(?)
        sim_name = mat_file.split("/")[-1][:-4]
        res_dict[sim_name] = res_df
    return res_dict


def find_changing_var_from_parameter_study(study_csv, res_dict):
    """
    Takes the study.csv and converts it into a dataframe.
    Then finds the parameter that is changing inside the parameter study and returns it
    :param study_csv:
    :param res_dict:
    :return:
    """

    study_df = pd.read_csv(study_csv, index_col=0)
    study_indexes = list(study_df.index.values)
    sim_names = list(res_dict.keys())
    if len(sim_names) <= 1:
        raise Exception("Please pass more than one simulation to the 'plot_from_df_looped' function or use the"
                        "single plot function")
    sim_names_to_drop = [x for x in study_indexes if x not in sim_names]
    study_df = study_df.drop(index=sim_names_to_drop)  # drop all sims of master.csv, that are'nt inside res_dict
    study_df = study_df[[i for i in study_df if len(set(study_df[i])) > 1]]
    study_df_reduced = study_df.dropna(axis=1, how="any")
    var0 = study_df.columns[0]

    return var0, study_df_reduced


def plot_from_df_looped(res_dict, dir_output, var_lst, y_label, plot_style, study_csv, savename_append='',
                        save_figs=False):
    """

    Creates a Matplotlib figure and plots the variables in 'var_lst'. Saves the figure in 'dir_output'

    :param study_csv:                   Path where the overview file of the simulation study is stored
    :param plot_style: str              plot style 1: one figure plots all variables of a single simulation
                                        plot style 2: one figure plots a single variable of all simulations
    :param res_dict: dictionary         dictionary that holds the Simulation names as keys and the corresponding
                                        variable data as a dataframe as values
    :param y_label: string
    :param var_lst: list                list of variables you want to plot
                                        -> should be formed with the "add_to_sublists" function
    :param dir_output: String:          path to output directory
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """
    fig_lst = []  # for return statement

    var0, study_df = find_changing_var_from_parameter_study(study_csv, res_dict)

    if plot_style == 'Single Simulation':  # Single Simulation, one line is one variable
        for sim_name in res_dict.keys():

            fig, ax = plt.subplots()
            sim_title = str(var0) + " = " + str(study_df.at[sim_name, var0])
            fig.suptitle(sim_title, fontsize=18)
            lines = []

            for var in var_lst:
                if "Simple" in str(var):
                    var_title = 'SimpleDistrict ' + ''.join(char for char in str(var) if char.isdigit())
                elif 'pipe' in str(var):
                    var_title = 'Pipe ' + ''.join(char for char in str(var) if char.isdigit())
                else:
                    var_title = str(var)
                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7, label=var_title)
                lines += line

            ax.set_ylabel(y_label)
            ax.set_xlabel("Time in Date")
            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0., ncol=2, fontsize=5)
            fig.autofmt_xdate(rotation=45, ha="center")

            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".pdf"))
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".png"), dpi=600)

    elif plot_style == 'Single Variable':  # Single Variable, one line is one simulation
        for var in var_lst:

            fig, ax = plt.subplots()

            if "Simple" in str(var):
                var_title = 'SimpleDistrict ' + ''.join(char for char in str(var) if char.isdigit())
            elif 'pipe' in str(var):
                var_title = 'Pipe ' + ''.join(char for char in str(var) if char.isdigit())
            else:
                var_title = str(var)

            fig.suptitle(var_title, fontsize=12)
            lines = []

            for sim_name in res_dict.keys():
                val0 = study_df.loc[sim_name][var0]
                # val1 = study_df._get_value(sim_name, var0)
                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7,
                               label=str(var0) + ' = ' + str(val0))
                lines += line

            ax.set_ylabel(y_label)
            ax.set_xlabel("Time in Date")
            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0., ncol=2, fontsize=5)
            fig.autofmt_xdate(rotation=45, ha="center")

            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".pdf"))
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".png"), dpi=600)

    return fig_lst


# streamlit renaming functions


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

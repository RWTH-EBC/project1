import pandas as pd
import os
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
from modelicares import SimRes
import fnmatch
import platform
from PIL import Image


def main():
    # ------------------------------ paths -----------------------------
    if platform.system() == 'Darwin':
        dir_sciebo = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    elif platform.system() == 'Windows':
        dir_sciebo = "D:/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    else:
        raise Exception("Unknown operating system")
    dir_models = dir_sciebo + "/models"
    dir_output = dir_sciebo + "/plots"

    changing_vars, study_df_reduced = reduce_and_update_study_df(dir_models)

    # --------------------- get a list of all variables --------------------------
    all_vars_lst = read_trajectory_names_folder(res_dir=dir_models, only_from_first_sim=True)

    # ---------------------------- making sublists ----------------------------
    all_sublists = {}
    # add_to_sublists(all_sublists, 'Mass Flow Supply Pipes', all_vars_lst, "senMasFlo.m_flow", "pipe", "R")
    # add_to_sublists(all_sublists, 'Mass Flow Return Pipes', all_vars_lst, "senMasFlo.m_flow", "R", "")
    # add_to_sublists(all_sublists, 'Supply and Return Temperatures', all_vars_lst, "senT", ".T", "")
    # add_to_sublists(all_sublists, 'Supply Variables', all_vars_lst, "networkModel.supply")
    add_to_sublists(all_sublists, 'Electrical_Power_Demand_Heatpump', all_vars_lst, 'heaPum.P')
    add_to_sublists(all_sublists, 'Return_Temp_Substations_after_del', all_vars_lst, 'del1.T')
    add_to_sublists(all_sublists, 'Return_Temp_Substations_before_del', all_vars_lst, 'senTem_return.T')
    add_to_sublists(all_sublists, 'Pressure_Drop_Substations', all_vars_lst, 'dpOut')
    add_to_sublists(all_sublists, 'Supply_Temp_Substation', all_vars_lst, 'senTem_supply.T')
    add_to_sublists(all_sublists, 'P_el_Substations', all_vars_lst, 'P_el')
    add_to_sublists(all_sublists, 'P_el_central_Pump', all_vars_lst, 'Supply.fan.P')
    add_to_sublists(all_sublists, 'P_el_central_heater', all_vars_lst, 'Supply.heater.Q_flow')

    # ---------------------- converting the mat files to dataframes, takes long! -----------------------
    res_dict = mat_files_filtered_to_dict(res_dir=dir_models, var_lst=all_vars_lst)

    # ------------------------------ test plotting ----------------------------------
    fig_lst = plot_from_df_looped(res_dict, dir_output, var_lst=all_sublists['P_el_central_Pump'],
                                  plot_style="Single Variable", dir_models=dir_models,
                                  y_label='P_el_central_Pump', save_figs=True)
    plt.show()

    # ------------------------------------------- Streamlit ------------------------------------------------------
    st.sidebar.title("Data Selection for Destest")
    show_map = st.checkbox('Show Destest Map')
    if show_map:
        destest_map = Image.open('/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing'
                                 '/uesgraph_destest_16_selfsized_jonas.png')
        st.image(destest_map, caption='Destest Network Layout', use_column_width=True)

    # choose a sublist of variables
    selected_sublist_description = st.sidebar.radio(label="Choose which sublist of variables to plot",
                                                    options=list(all_sublists.keys()))
    # choose a plot style
    selected_plot_style = st.sidebar.radio(label="Choose Plotstyle", options=['Single Variable', 'Single Simulation'])

    # choose Variables from selected sublist
    selected_variables_from_sublist = st.sidebar.multiselect(
        label="If you plot a single Variable per Plot, choose which variable to plot from the chosen sublist",
        options=all_sublists[selected_sublist_description],
        default=all_sublists[selected_sublist_description][-2:])

    # choose Simulations from the res_dict keys and save them as a new plot_res_dict
    selected_simulation_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single simulation per Plot, choose which simulation to plot",
        options=list(res_dict.keys()),
        default=list(res_dict.keys())[0:5])
    plot_res_dict = {}
    for sim in list(res_dict.keys()):
        if sim in selected_simulation_from_sublist:
            plot_res_dict.update({sim: res_dict[sim]})

    show_changing_vars = st.checkbox("Show changing variables")
    if show_changing_vars:
        st.write(study_df_reduced)

    # One Plot displays one Single Variable, one line is one simulation -> Dymola can't do this

    fig_lst = plot_from_df_looped(
        res_dict=plot_res_dict,
        dir_output=dir_output,
        plot_style=selected_plot_style,
        var_lst=selected_variables_from_sublist,
        dir_models=dir_models,
        y_label=selected_sublist_description,
        save_figs=False)

    for fig in fig_lst:
        st.pyplot(fig)


def create_streamlit_dashboard_simple(all_sublists, res_dict, dir_output, dir_models):
    """
    Creates a Streamlit Dashboard. Straemlit dashboards have to be started by the Terminal. To run streamlit,
    navigate inside the Terminal to the folder where this very file is stored, then enter
    "streamlit run Explore_matfile.py". You Can also stay at your top level folder and enter the full path to this file.
    It is recommended to create 2 Terminal shortcuts like:
    1) 'cau' -> conda activate uegraphs_py36
    2) 'stem' -> streamlit run D://...../Explore_matfile.py

    :param all_sublists:
    :param res_dict:
    :param dir_output:
    :param dir_models:
    :return:
    """
    st.sidebar.title("Data Selection for Destest")

    # choose a sublist of variables from all the sublists inside the all_sublists dictionary
    selected_sublist_description = st.sidebar.radio(label="Choose which sublist of variables to plot",
                                                    options=list(all_sublists.keys()))

    # choose a plot style
    selected_plot_style = 'Single Variable'

    # choose a Variable inside the selected sublist of variables
    selected_variables_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single Variable per Plot, choose which variable to plot from the chosen sublist",
        options=all_sublists[selected_sublist_description])

    # One Plot displays one Single Variable, one line is one simulation -> Dymola can't do this
    if selected_plot_style == 'Single Variable':
        vars_to_plot = selected_variables_from_sublist
        fig_lst = plot_from_df_looped_sns(res_dict, dir_output, plot_style="Single Variable", var_lst=vars_to_plot,
                                          dir_models=dir_models, y_label=selected_sublist_description)
        for fig in fig_lst:
            plt.gcf()
            st.pyplot(fig)


def create_streamlit_dashboard(all_sublists, res_dict, dir_output, dir_models):
    """
    Creates a Streamlit Dashboard. Straemlit dashboards have to be started by the Terminal. To run streamlit,
    navigate inside the Terminal to the folder where this very file is stored, then enter
    "streamlit run Explore_matfile.py". You Can also stay at your top level folder and enter the full path to this file.
    It is recommended to create 2 Terminal shortcuts like:
    1) 'cau' -> conda activate uegraphs_py36
    2) 'stem' -> streamlit run D://...../Explore_matfile.py

    :param all_sublists:
    :param res_dict:
    :param dir_output:
    :param dir_models:
    :return:
    """
    st.sidebar.title("Data Selection for Destest")
    show_map = st.checkbox('Show Destest Map')
    if show_map:
        destest_map = Image.open('/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing'
                                 '/uesgraph_destest_16_selfsized_jonas.png')
        st.image(destest_map, caption='Destest Network Layout', use_column_width=True)

    # choose a sublist of variables
    selected_sublist_description = st.sidebar.radio(label="Choose which sublist of variables to plot",
                                                    options=list(all_sublists.keys()))

    # choose a plot style
    selected_plot_style = st.sidebar.radio(label="Choose Plotstyle", options=['Single Simulation', 'Single Variable'])

    # choose Variables from selected sublist
    selected_variables_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single Variable per Plot, choose which variable to plot from the chosen sublist",
        options=all_sublists[selected_sublist_description],
        default=all_sublists[selected_sublist_description][-2:])

    # choose Simulations from the res_dict keys and save them as a new plot_res_dict
    selected_simulation_from_sublist = st.sidebar.multiselect(
        label="If you chose to plot a single simulation per Plot, choose which simulation to plot",
        options=list(res_dict.keys()),
        default=list(res_dict.keys())[0:5])
    plot_res_dict = {}
    for sim in list(res_dict.keys()):
        if sim in selected_simulation_from_sublist:
            plot_res_dict.update({sim: res_dict[sim]})

    study_df_reduced = reduce_and_update_study_df(dir_models)

    show_changing_vars = st.checkbox("Show changing variables")
    if show_changing_vars:
        st.write(study_df_reduced)

    # One Plot displays one Single Variable, one line is one simulation -> Dymola can't do this

    fig_lst = plot_from_df_looped(
        res_dict=plot_res_dict,
        dir_output=dir_output,
        plot_style=selected_plot_style,
        var_lst=selected_variables_from_sublist,
        dir_models=dir_models,
        y_label=selected_sublist_description)

    for fig in fig_lst:
        st.pyplot(fig)


def read_trajectory_names_folder(res_dir, only_from_first_sim=True):
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

    return sublist


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

    if len(sublist) > 0:
        master_sublists_dict.update({description: sublist})
    else:
        raise Exception("the Created Sublist {} has no entries,"
                        "check the input signals of the add_to_sublists function!".format(description))

    return {description: sublist}


def matfile_to_df(res_path, var_lst, output_interval="hours"):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. The variables given in the var_lst list
    are then converted to a pandas dataframe.   (... further explanation ...) The dataframe is then returned.
    :param res_path:    str:        Path to the .mat result file
    :param var_lst:     list:       variable names
    :param output_interval:         specify if your output interval is in seconds or hours.
                                    Should be the same as inside your uesgraphs create_model function.
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


@st.cache(persist=True)
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


def reduce_and_update_study_df(dir_models, model_prefix="Destest_Jonas", update_csv_files=True):
    """
    Takes the study.csv and converts it into a dataframe.
    Then finds the parameter that is changing inside the parameter study and returns it,
    as well as a reduced version of the study.csv, which only contains the variable names that are changing.
    :param dir_models:          csv that holds the simulation study info
    :param model_prefix:        search string for the model names
    :param update_csv_files:    decide to update the csv files after sims are deleted
    :return:
    """
    dir_models = Path(dir_models)
    study_csv = dir_models/"study.csv"
    curr_study_csv = dir_models / "curr_study.csv"

    # if theres no curr_study_csv yet, coply the study.csv
    if not curr_study_csv.is_file():
        shutil.copy(study_csv, curr_study_csv)

    # study.csv was created from automated_network_simulation, should always stay the same!
    study_df = pd.read_csv(study_csv, index_col=0)
    sims_study_df = list(study_df.index.values)

    curr_study_df = pd.read_csv(curr_study_csv, index_col=0)
    sims_curr_study_df = list(curr_study_df.index.values)

    # udate curr_study_df

    # 1) remove sims from curr_study_df that are in curr_study_df but not in the folder
    curr_sims = [child.name for child in dir_models.iterdir() if model_prefix in child.name]
    sims_to_remove = [sim for sim in sims_curr_study_df if sim not in curr_sims]
    curr_study_df = curr_study_df.drop(index=sims_to_remove)

    # 2) add sims to curr_study_df that are in the folder but not in curr_study_df,
    # condition: they have to be in the original study_df

    # # all_deleted_sims = [sim for sim in all_sims if sim not in curr_sims]
    # deleted_sims = [sim for sim in sims_study_df if sim not in sims_curr_study_df]
    #
    # if len(sims_curr_study_df) > len(curr_sims):
    #     curr_study_df = curr_study_df.drop(index=deleted_sims)  # drop deleted sims
    # elif len(sims_curr_study_df) < len(curr_sims):
    #     pass
    #     # Todo: re add re_added_sim with the info from the original study_df
    #     # re_added_sims = [sim for sim in curr_sims if sim not in curr_sims_in_df]
    #     # not_re_added_sim = [sim for sim in curr_sims if sim in curr_sims_in_df]
    #     # re_added_df = study_df.drop(index=not_re_added_sim)
    #     # curr_study_df = curr_study_df.append(re_added_df)

    # reduce df: remove columns that have only constants
    curr_study_df_reduced = reduce_df(curr_study_df)

    # make list [Sim 1, Sim 2..], add it as a new column to the dataframe and put at first colum position
    curr_study_df_reduced = curr_study_df_reduced.sort_index()  # for assigning short sim names in ascending order
    short_sim_names = ['Sim ' + str(x + 1) for x in range(len(curr_study_df.index))]
    curr_study_df_reduced['short_sim_name'] = short_sim_names
    cols = ['short_sim_name'] + [col for col in curr_study_df_reduced if col != 'short_sim_name']
    curr_study_df_reduced = curr_study_df_reduced[cols]

    # save new and old version as csv files
    curr_study_df_reduced.to_csv(dir_models/"curr_study.csv")

    changing_vars = [var for var in curr_study_df_reduced.columns if 'short_sim_name' not in var]

    return changing_vars, curr_study_df_reduced


def reduce_df(input_df):
    """
    Takes a Dataframe and returns a reduced version, which only contains the columns where the variables are changing.
    :param      input_df:           Dataframe that holds Simulation Names as indexes and variables as columns
    :return:    input_df_reduced:   Dataframe, reduced number of columns
    """
    if len(input_df.index) <= 1:
        raise Exception("Cant find changing vars from a Dataframe that only has one row")
    input_df = input_df[[i for i in input_df if len(set(input_df[i])) > 1]]  # drop constant columns
    input_df = input_df.dropna(axis=1, how="any")  # drops all columns with only NaN's
    input_df_reduced = input_df.sort_index()    # sort for assigning short sim names in ascending order

    return input_df_reduced


def plot_from_df_looped(res_dict, dir_output, var_lst, y_label, plot_style, dir_models, savename_append='',
                        save_figs=False):
    """
    Creates a Matplotlib figure and plots the variables in 'var_lst'. Saves the figure in 'dir_output'

    :param dir_models:                  Folder where the overview file of the simulation study is stored
    :param plot_style: str              plot style 1: one figure plots all variables of a single simulation
                                        plot style 2: one figure plots a single variable of all simulations
    :param res_dict: dictionary         dictionary that holds the Simulation names as keys and the corresponding
                                        variable data as a dataframe as values
    :param y_label: string              Label of the y-Axis
    :param var_lst: list                list of variables you want to plot
                                        -> should be a subset of the variables inside res_dict!
                                        -> should be formed with the "add_to_sublists" function
    :param dir_output: String:          path to output directory
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """
    fig_lst = []  # for return statement
    plt.style.use("/Users/jonasgrossmann/git_repos/matplolib-style/ebc.paper.mplstyle")
    sns.set()
    sns.set_context("paper")
    changing_vars, study_df_reduced = reduce_and_update_study_df(dir_models)

    # -------------- 1 Simulation per Plot, one line is one variable, this is what Dymola CAN do ----------------
    if plot_style == 'Single Simulation':
        for sim_name in res_dict.keys():    # one sim per plot

            # the figure title is derived by the parameters that are changing inside the simulation study
            sim_title_long = ''
            for var_x in changing_vars:
                sim_title_x = str(var_x) + " = " + str(study_df_reduced.at[sim_name, var_x]) + " "
                sim_title_long += sim_title_x
            sim_title_short = study_df_reduced.at[sim_name, 'short_sim_name']

            fig, ax = plt.subplots()
            # fig.suptitle(sim_title_long, y=1.1, fontsize=10)
            plt.title(sim_title_short, fontsize=12, y=1)
            lines = []

            for var in var_lst:     # legend and line naming
                if "Simple" in str(var):
                    var_title = 'SimpleDistrict ' + ''.join(char for char in str(var) if char.isdigit())
                elif 'pipe' in str(var):
                    var_title = 'Pipe ' + ''.join(char for char in str(var) if char.isdigit())
                else:
                    var_title = str(var)

                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7, label=var_title)
                lines += line

            ax.set_ylabel(y_label.replace("_", " "))
            ax.set_xlabel("Time in Date")

            date_form = mdates.DateFormatter('%b')
            date_loc = mdates.MonthLocator()
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(date_loc)

            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0.2, ncol=1, fontsize=8)
            fig.autofmt_xdate(rotation=45, ha="center")

            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".pdf"))
                # fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".png"), dpi=600)

    # ------------ Single Variable, one line is one simulation, this is what Dymola CANT do -------------------
    elif plot_style == 'Single Variable':
        for var in var_lst:

            fig, ax = plt.subplots()

            # var_title = name of the variable
            if "Simple" in str(var):
                var_title = 'SimpleDistrict ' + ''.join(char for char in str(var) if char.isdigit())
            elif 'pipe' in str(var):
                var_title = 'Pipe ' + ''.join(char for char in str(var) if char.isdigit())
            else:
                var_title = str(var)

            plt.title(var_title, fontsize=12, y=1)
            lines = []

            # this is for naming the legend and drawing the lines
            for sim_name in res_dict.keys():

                line_label = ''
                for var_x in changing_vars:
                    val_x = study_df_reduced.loc[sim_name][var_x]
                    line_label_x = str(var_x) + ' = ' + str(val_x) + '\n'
                    line_label += line_label_x

                line_label_short = study_df_reduced.at[sim_name, 'short_sim_name']

                line = ax.plot(res_dict[sim_name][var].resample("D").mean(), linewidth=0.7, label=line_label_short)
                lines += line

            ax.set_ylabel(y_label.replace("_", " "))
            ax.set_xlabel("Time in Date")

            date_form = mdates.DateFormatter('%b')
            date_loc = mdates.MonthLocator()
            ax.xaxis.set_major_formatter(date_form)
            ax.xaxis.set_major_locator(date_loc)

            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0.5, ncol=2, fontsize=8)
            fig.autofmt_xdate(rotation=45, ha="center")

            fig_lst.append(fig)

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".pdf"))
                # fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".png"), dpi=600)

    return fig_lst


def plot_from_df_looped_sns(res_dict, dir_output, var_lst, y_label, plot_style, dir_models, savename_append='',
                            save_figs=False):
    """
    Creates a Seaborn figure and plots the variables in 'var_lst'. Saves the figure in 'dir_output'

    :param dir_models:                  Folder where the overview file of the simulation study is stored
    :param plot_style: str              plot style 1: one figure plots all variables of a single simulation
                                        plot style 2: one figure plots a single variable of all simulations
    :param res_dict: dictionary         dictionary that holds the Simulation names as keys and the corresponding
                                        variable data as a dataframe as values
    :param y_label: string              Label of the y-Axis
    :param var_lst: list                list of variables you want to plot
                                        -> should be a subset of the variables inside res_dict!
                                        -> should be formed with the "add_to_sublists" function
    :param dir_output: String:          path to output directory
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """

    # fig_lst = plot_from_df_looped_sns(
    #     res_dict=plot_res_dict,
    #     dir_output=dir_output,
    #     plot_style=selected_plot_style,
    #     var_lst=selected_variables_from_sublist,
    #     study_csv=study_csv,
    #     y_label=selected_sublist_description)

    fig_lst = []  # for return statement
    long_sim_titles = False
    plt.style.use("/Users/jonasgrossmann/git_repos/matplolib-style/ebc.paper.mplstyle")
    sns.set()
    sns.set_context("paper", rc={"lines.linewidth": 0.5})  # style of the plot
    # sns.set_palette(sns.color_palette(eonerc_colors))     # eventually use costum colours?
    changing_vars, study_df_reduced = reduce_and_update_study_df(dir_models)

    # -------------- 1 Simulation per Plot, one line is one variable, this is what Dymola CAN do ----------------
    if plot_style == 'Single Simulation':
        sim_idx_dict = {}

        fig, ax = plt.figure()
        # fig.suptitle(sim_title, fontsize=10)
        lines = []

        for sim_idx, sim_name in enumerate(res_dict.keys(), start=1):

            # the figure title is derived by the parameters that are changing inside the simulation study
            sim_title_long = ''
            for var_x in changing_vars:
                sim_title_x = str(var_x) + " = " + str(study_df_reduced.at[sim_name, var_x]) + " "
                sim_title_long += sim_title_x
            sim_title_short = 'Sim ' + str(sim_idx)
            sim_idx_dict[str(sim_idx)] = sim_title_long
            if long_sim_titles:
                sim_title = sim_title_long
            else:
                sim_title = sim_title_short

            # this is for the legend naming
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
                # fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".png"), dpi=600)

    # ------------ Single Variable, one line is one simulation, this is what Dymola CANT do -------------------

    elif plot_style == 'Single Variable':
        for var in var_lst:

            # ----- Data Selection -------
            data_df = pd.DataFrame()    # stores all the lines as columns
            sim_idx_dict = {}
            for sim_idx, sim_name in enumerate(res_dict.keys(), start=1):

                # ----- Legend Names for each plotted Line ------
                sim_title_long = ''
                for var_x in changing_vars:
                    sim_title_x = str(var_x) + " = " + str(study_df_reduced.at[sim_name, var_x]) + " "
                    sim_title_long += sim_title_x
                sim_title_short = 'Sim ' + str(sim_idx)

                sim_idx_dict[str(sim_idx)] = sim_title_long

                if long_sim_titles:
                    sim_title = sim_title_long
                else:
                    sim_title = sim_title_short

                # df of one sim, take the column with 'var' and append it with the 'sim_title' to the 'data_df'
                sim_df = res_dict[sim_name]
                var_df = sim_df[[var]]
                var_df = var_df.rename(columns={var: sim_title})  # rename the column
                data_df[sim_title] = var_df[sim_title]  # add to the data_df

            # ----- Plot ------
            sns.lineplot(data=data_df, dashes=False)
            # legend = plt.legend(bbox_to_anchor=(0, 1.02, 1, 1), loc="lower left",
            #            mode="expand", borderaxespad=0, ncol=2)
            legend = plt.legend(loc='best', ncol=8, fontsize=5)
            plt.title(var)
            # plt.tight_layout()
            fig = plt.gcf()  # get current figure
            fig_lst.append(fig)

            # ----- Save Figure ------
            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".pdf"))
                # fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".png"), dpi=600)

    return fig_lst


# unused function
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

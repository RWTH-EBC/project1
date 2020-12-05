import pandas as pd
import os
from pathlib import Path
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import streamlit as st
from modelicares import SimRes
import fnmatch
import platform
from PIL import Image
from datetime import datetime

# from wp_3_2_destest.Network.NetworkSizing.destest_uesgraphs_jonas import import_demands_from_demgen  # no streamlit

font_size = 11
line_width = 0.8
fig_width = (155 / 25.4)  # 1inch=25.4mm
fig_height = (fig_width / (16 / 9)) * 0.8
figsize = (fig_width, fig_height)
marker_size = 10
alpha = 1
# Change general parameters
# https://matplotlib.org/api/font_manager_api.html
# https://matplotlib.org/3.1.1/tutorials/text/text_intro.html
# https://stackoverflow.com/questions/21933187/how-to-change-legend-fontname-in-matplotlib
# https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html
# ändert nicht die schriftart bei Seaborn ....
rcParams_dict = {"font.size": font_size,
                 "font.family": 'sans-serif',
                 "font.sans-serif": 'Heuristica',
                 "lines.linewidth": line_width,
                 "axes.linewidth": line_width,
                 "lines.markersize": marker_size,
                 'legend.fontsize': font_size,
                 'xtick.labelsize': font_size,
                 'ytick.labelsize': font_size,
                 'axes.labelsize': font_size,
                 'axes.titlesize': font_size}
mpl.rcParams.update(rcParams_dict)
mpl.rc('font', family='Heuristica')
sns.set_context(rc=rcParams_dict)


def main():
    # ------------------------------ paths -----------------------------
    if platform.system() == 'Darwin':
        dir_sciebo = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
        dir_micha = "/Users/jonasgrossmann/sciebo/MA_Jonas"
    elif platform.system() == 'Windows':
        dir_sciebo = "D:/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
        dir_micha = "D:/mma-jgr/sciebo-folder/MA_Jonas"
    else:
        raise Exception("Unknown operating system")

    dir_output = dir_sciebo + "/plots"
    dir_models = dir_sciebo + "/models"
    dir_case1_some = dir_sciebo + "/saved_models/case1_some"
    dir_case1 = dir_sciebo + "/saved_models/case1"
    dir_case1C = dir_micha + "/saved_models/case1C"
    dir_basecase = dir_micha + "/saved_models/basecase"
    dir_case1B_1C = dir_micha + "/saved_models/case1B_1C"
    dir_case1A_BC = dir_micha + "/saved_models/case1A_BC"
    # dir_basecase = dir_micha + "/saved_models/basecase_dp"

    dir_case1C = dir_micha + "/final_models/case_1C"
    dir_case1B = dir_micha + "/final_models/case_1B"
    dir_case_BC_1A = dir_micha + "/final_models/Case_BC_1A"
    dir_case_1B_1C = dir_micha + "/final_models/Case_1B_1C"

    dir_models = dir_case_BC_1A

    study_df = load_study_df(dir_models=dir_models)
    changing_vars, study_df_reduced = reduce_and_update_study_df(dir_models=dir_models)

    # ------- compute KPIs -------
    # changing_vars_bc, study_df_reduced_bc = compute_all_KPIs(dir_sciebo=dir_sciebo,
    #                                                          dir_models=dir_models, case="CaseBase")
    #
    # changing_vars_c1, study_df_reduced_c1 = compute_all_KPIs(dir_sciebo=dir_sciebo,
    #                                                          dir_models=dir_models, case="Case1")

    # study_df_reduced = study_df_reduced_bc
    # changing_vars = changing_vars_c1

    # write_const_to_study_csv(dir_models=dir_models, const_name_end='cor.del.dh', column_name='dh')
    study_df_reduced = pd.read_csv(dir_models + "/study_reduced.csv", index_col=0)

    # --------------------- get a list of all variables --------------------------
    all_vars_lst = read_trajectory_names_folder(res_dir=dir_models, only_from_first_sim=True)

    # ---------------------------- making sublists, Case 1 ----------------------------
    all_sublists = {}
    # add_to_sublists(all_sublists, 'Fluid_Velocity', all_vars_lst, sig1=".cor.del.v", sig2="pipe", anti_sig1="R")
    # add_to_sublists(all_sublists, 'Return_Temp_Substations_after_del', all_vars_lst, sig1='del1.T')
    # add_to_sublists(all_sublists, 'Return_Temp_Substations_before_del', all_vars_lst, sig1='senTem_return.T')
    # add_to_sublists(all_sublists, 'Pressure_Drop_Substations', all_vars_lst, sig1='dpOut')
    # add_to_sublists(all_sublists, 'Supply_Temp_Substation', all_vars_lst, sig1='senTem_supply.T')
    # add_to_sublists(all_sublists, 'P_el_Substations', all_vars_lst, end='P_el.y')
    # add_to_sublists(all_sublists, 'P_el_central_Pump', all_vars_lst, sig1='Supply.fan.P')
    # to_comb_sublist(all_sublists, 'Supp_and_Ret_Temps', all_vars_lst, 'senTem_supply.T', 'del1.T')
    # demands_lst = ['networkModel.SimpleDistrict_7dhw_input', 'networkModel.SimpleDistrict_7heat_input',
    #                'networkModel.SimpleDistrict_7cold_input']
    # add_to_sublists(all_sublists, 'Demands', demands_lst)

    # ---------------------------- making sublists, BaseCase ----------------------------

    add_to_sublists(all_sublists, 'P_el_central_heater', all_vars_lst, 'Supply.heater.Q_flow')  # only in BaseCase

    # ---- very specific lists --------

    pipes_lst = ["networkModel.pipe1001to1005.cor.del.v", 'networkModel.pipe1022to1025R.cor.del.v',
                 'networkModel.pipe1005to1020.cor.del.v']
    pipes_names_lst = ['Pipe 1-5', 'Pipe 22-25', 'Pipe 5-20']
    pipes_dict = {
        "networkModel.pipe1001to1005.cor.del.v": 'Pipe 1-5',
        'networkModel.pipe1022to1025R.cor.del.v': 'Pipe 22-25',
        'networkModel.pipe1005to1020.cor.del.v': 'Pipe 5-20'
    }

    add_to_sublists(all_sublists, 'Fluid_Velocity_in_m/s', pipes_lst)

    # make one list that inclesed all the sublsits variables
    all_selected_vars_lst = []
    for var_lst in all_sublists.values():
        all_selected_vars_lst += var_lst

    # --------- instead of lists, make dicts: -----------

    pipes_dict = {
        "networkModel.pipe1001to1005.cor.del.v": 'Pipe 1-5',
        'networkModel.pipe1022to1025R.cor.del.v': 'Pipe 22-25',
        'networkModel.pipe1005to1020.cor.del.v': 'Pipe 5-20'
    }


    master_vars_dict = {

    }


    # ---------------------- converting the mat files to dataframes, takes long! -----------------------
    res_dict = mat_files_filtered_to_dict(res_dir=dir_models, var_lst=all_selected_vars_lst)

    # ------------------------------ test plotting ----------------------------------
    # fig_lst = plot_from_df_looped(res_dict, dir_output, var_lst=all_sublists['Fluid_Velocity_2'],
    #                               plot_style="Single Variable", dir_models=dir_models,
    #                               y_label='Fluid_Velocity', save_figs=True,
    #                               start_date='2019-12-11', end_date='2019-12-12')
    # plt.show()

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

    start_date, end_date = st.slider(
        label="Choose start and end date:",
        min_value=datetime(2019, 1, 1, 0, 0),
        max_value=datetime(2020, 1, 1, 0, 0),
        value=(datetime(2019, 1, 1), datetime(2020, 1, 1))
    )

    resample_delta = st.sidebar.radio(label="Choose a resample delta",
                                      options=['600S', '1800S', 'H', '6H', 'D'],
                                      )

    fig_lst = plot_from_df_looped(
        res_dict=plot_res_dict,
        dir_output=dir_output,
        plot_style=selected_plot_style,
        var_lst=selected_variables_from_sublist,
        dir_models=dir_models,
        y_label=selected_sublist_description,
        start_date=start_date, end_date=end_date,
        resample_delta=resample_delta,
        save_figs=True)

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


def read_constants_names_folder(res_dir, only_from_first_sim=True):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. Then, all variable names that are
    considered constants are saved in a list.

    :param only_from_first_sim: decide if you want to take the constant names only from the first simulation of a
                                parameter study or of all simulations. It makes sense to take the constants of all
                                simulations, if the variable names change. For example, when using different substation
                                or supply models within the same simulation study.
    :param res_dir:
    :return all_trajectories_lst:   list:   names of all variables that are trajectories
        """
    res_all_lst = find("*.mat", res_dir)

    if only_from_first_sim:
        res_path = res_all_lst[0]
        print("Converting one .mat File to a SimRes instance to get the constants names ...")
        sim = SimRes(fname=res_path)
        all_constants_lst = sim.get_constants()  # all_trajectory_names is list of variables that are trajectories
        print("There are " + str(len(all_constants_lst)) + " constants in the first results file.")
    else:
        all_constants_lst = []
        print("Converting the .mat Files to SimRes instances to get all constants names ...")
        for res_path in res_all_lst:
            sim = SimRes(fname=res_path)
            sim_constants_lst = sim.get_constants()
            all_constants_lst.extend(sim_constants_lst)
        all_constants_lst = list(set(all_constants_lst))  # drops duplicates

    return all_constants_lst


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


def make_sublist(input_var_lst, sig1="", end="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal words given. This is useful for plotting a specific
    subset of variables. Returns the list together with its description as a dictionary.

    :param input_var_lst: list      input list
    :param sig1: String             String that should be included in the sublists results
                                    Examples: pipe, demand, networkModel.supply
    :param end: String              End String of the Variable. Examples: .vol.T,
    :param anti_sig1: String        String that shouldn't be included in th sublists results. Examples: R,
    :return sub_var_lst1: list      sublist
    """
    sublist = [i for i in input_var_lst if sig1 in i]
    if end:
        sublist = [i for i in sublist if i.endswith(end)]
    if anti_sig1:
        sublist = [i for i in sublist if anti_sig1 not in i]

    return sublist


def to_comb_sublist(master_sublists_dict, description, input_var_lst, sig1, sig2):
    """
    Combines to sublists to a single sublist. Returns the list together with its description as a dictionary.

    :param master_sublists_dict:    dictionary that stores all the sublists as values and their description as keys
    :param description: String      description of the variables inside the output sublist
    :param input_var_lst: list      input list
    :param sig1: String             String that should be included in the sublists results
                                    Examples: pipe, demand, networkModel.supply
    :param sig2: String             String that should be included in the sublists results. Examples: .vol.T,
    :return sub_var_lst1: list      sublist
    """
    sublist1 = [i for i in input_var_lst if sig1 in i]
    sublist2 = [i for i in input_var_lst if sig2 in i]

    sublist = sublist1 + sublist2

    if len(sublist) > 0:
        master_sublists_dict.update({description: sublist})
    else:
        raise Exception("the Created Sublist {} has no entries,"
                        "check the input signals of the add_to_sublists function!".format(description))

    return {description: sublist}


def add_to_sublists(master_sublists_dict, description, input_var_lst, end="", sig1="", sig2="", anti_sig1=""):
    """
    Makes a sublist of a list, depending on the signal words given. This is useful for plotting a specific
    subset of variables. Returns the list together with its description as a key:value pair. Adds this pair to the
    master_sublists_dict that contains all those key:value pairs.

    :param master_sublists_dict:    dictionary that stores all the sublists as values and their description as keys
    :param description: String      description of the variables inside the output sublist
    :param input_var_lst: list      input list
    :param end: String              String with which the variable ends, e.g ".fan.P"
    :param sig1: String             String that should be included in the sublists results
                                    Examples: pipe, demand, networkModel.supply
    :param sig2: String             String that should be included in the sublists results. Examples: .vol.T,
    :param anti_sig1: String        String that shouldn't be included in th sublists results. Examples: R,
    :return sub_var_lst1: list      sublist
    """
    sublist = [i for i in input_var_lst if sig1 in i]
    if sig2:
        sublist = [i for i in sublist if sig2 in i]
    if anti_sig1:
        sublist = [i for i in sublist if anti_sig1 not in i]
    if end:
        sublist = [i for i in sublist if i.endswith(end)]

    if len(sublist) > 0:
        master_sublists_dict.update({description: sublist})
    else:
        raise Exception("the Created Sublist {} has no entries, "
                        "check the input signals of the add_to_sublists function!".format(description))

    return {description: sublist}


@st.cache()
def matfile_to_df(res_path, var_lst, output_interval="seconds", s_step=600):
    """
    Takes the .mat file from the result path and saves it as a SimRes Instance. The variables given in the var_lst list
    are then converted to a pandas dataframe.   (... further explanation ...) The dataframe is then returned.
    :param res_path:    str:        Path to the .mat result file
    :param var_lst:     list:       variable names
    :param output_interval:         specify if your output interval is in seconds or hours.
                                    Should be the same as inside your uesgraphs create_model function.
    :param s_step:      int:        timestep in seconds, f.e.: 600s=10min, 900s=15min
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
        res_df = res_df[res_df.index.isin(range(0, 31536000, s_step))]
        res_df.index = res_df.index.astype(int)
        res_df.index = pd.to_datetime(res_df.index, unit="s", origin="2019")
    elif output_interval == "hours":
        res_df = res_df[res_df.index.isin(range(0, 8760, 1))]
        res_df.index = res_df.index.astype(int)
        # res_df.index = pd.to_datetime(res_df.index, unit="h", origin="2019") #-> to_datetime has no Unit "Hour"
        res_df.index = pd.Timestamp('2019-01-01') + pd.to_timedelta(res_df.index, unit='H')
    else:
        raise Exception("Unknown Output Interval")
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
        res_df = matfile_to_df(res_path=mat_file, var_lst=var_lst, output_interval="seconds")
        if save_to_csv:
            print("Converting the Dataframe to a CSV ... This could take some minutes!")
            savename_csv = mat_file.split("/")[-1][:-4], savename_append
            res_df.to_csv(savename_csv)  # prints an overview over the csv and saves it to the current folder(?)
        sim_name = mat_file.split("/")[-1][:-4]
        res_dict[sim_name] = res_df
    return res_dict


def load_study_df(dir_models):
    dir_models = Path(dir_models)
    study_csv = Path(dir_models / 'study.csv')

    # if theres already a study.csv, just read it
    if study_csv.is_file():
        study_df = pd.read_csv(study_csv, index_col=0)

    # if theres no study.csv, we have to make one from the overview files of the individual sims
    else:
        overview_files = find("*overview.csv", dir_models)
        if len(overview_files) == 0:
            raise Exception("No overview.csv files were found in the models folder")
        else:
            shutil.copy(overview_files[0], dir_models)
            first_csv_name = Path(overview_files[0]).name
            first_study_csv = Path(dir_models / first_csv_name)
            first_study_csv.rename(study_csv)
            del overview_files[0]  # for not appending it again

            # this study df consists only of the first overview file right now
            study_df = pd.read_csv(study_csv, index_col=0)

            # append all the other overview files to the study_df
            for overview_file in overview_files:
                overview_df = pd.read_csv(overview_file, index_col=0)
                study_df = pd.concat([study_df, overview_df], axis='rows')
                study_df.drop_duplicates()
                study_df.sort_index(axis='index', ascending=False, inplace=True)

            # save the study_df as the new study.csv
            study_df.to_csv(study_csv)

    return study_df


def reduce_and_update_study_df(dir_models, model_prefix="Case"):
    """
    Takes the study.csv and converts it into a dataframe.
    Then finds the parameter that is changing inside the parameter study and returns it,
    as well as a reduced version of the study.csv, which only contains the variable names that are changing.
    :param dir_models:          csv that holds the simulation study info
    :param model_prefix:        search string for the model names
    :return:
    """
    dir_models = Path(dir_models)
    study_df = pd.read_csv(dir_models / "study.csv", index_col=0)

    sims_study_df = list(study_df.index.values)

    # Update: remove sims from study_df that are not anymore in the folder (-> deleted)
    curr_sims = [child.name for child in dir_models.iterdir() if model_prefix in child.name]
    sims_to_remove = [sim for sim in sims_study_df if sim not in curr_sims]
    study_df = study_df.drop(index=sims_to_remove)

    # reduce df: remove columns that have only constants
    study_df_reduced = reduce_df(study_df)

    # make list [Sim 1, Sim 2..], add it as a new column to the dataframe and put at first column position
    study_df_reduced = study_df_reduced.sort_index()  # for assigning short sim names in ascending order
    short_sim_names = ['Sim ' + str(x + 1) for x in range(len(study_df.index))]
    study_df_reduced['short_sim_name'] = short_sim_names
    cols = ['short_sim_name'] + [col for col in study_df_reduced if col != 'short_sim_name']
    study_df_reduced = study_df_reduced[cols]

    # Update the csv's
    study_df_reduced.to_csv(dir_models / "study_reduced.csv")
    study_df.to_csv(dir_models / "study.csv")

    # entries of the study.csv that shouldnt be considered a variable
    no_vars = ['short_sim_name', 'W_tot_kWh', 'GWI', 'OP', 'W_Central_Pump_kWh',
               'W_Central_HP_kWh', 'OP_tot_HKW', 'OP_tot_BHKW', 'OP_tot_KKW',
               'GWI_tot_KKW', 'GWI_tot_BHKW', 'GWI_tot_HKW', 'GWI_HKW', 'GWI_BHKW',
               'GWI_KKW', 'GWI_Central_Pump', 'Q_tot_kWh', 'demand__Q_flow_nominal',
               'demand__Q_flow_input', 'save_name']

    changing_vars = [var for var in study_df_reduced.columns if var not in no_vars]

    return changing_vars, study_df_reduced


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
    input_df_reduced = input_df.sort_index()  # sort for assigning short sim names in ascending order

    return input_df_reduced


def plot_from_df_looped(res_dict, dir_output, var_lst, y_label, plot_style, dir_models, resample_delta,
                        start_date='2019-01-01', end_date='2020-01-01', savename_append='', save_figs=False):
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
    :param start_date:  String:         Start Date to plot vars in Timestamp-Format, e.g. 2019-01-01
    :param end_date:    String:         End Date to plot vars in Timestamp-Format, e.g. 2019-02-01
    :param savename_append: String:     Name to save the figures
    :param save_figs: Boolean:          decision to save figures to pdf and png
    :return: fig_lst: list:             list that contains all created figures
    """
    fig_lst = []  # for return statement
    plt.style.use("/Users/jonasgrossmann/git_repos/matplolib-style/ebc.paper.mplstyle")
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    changing_vars, study_df_reduced = reduce_and_update_study_df(dir_models)

    # -------------- 1 Simulation per Plot, one line is one variable, this is what Dymola CAN do ----------------
    if plot_style == 'Single Simulation':
        for sim_name in res_dict.keys():  # one sim per plot

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

                line = ax.plot(res_dict[sim_name][var][start_date:end_date].resample(resample_delta).mean(),
                               linewidth=0.7, label=var_title)
                lines += line

            ax.set_ylabel(y_label)

            # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
            formatter.zero_formats = [''] + formatter.formats[:-1]
            formatter.zero_formats[3] = '%d-%b'
            formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M', ]
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0.2, ncol=1, fontsize=12)

            fig_lst.append(fig)
            plt.show()

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

            # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)
            formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
            formatter.zero_formats = [''] + formatter.formats[:-1]
            formatter.zero_formats[3] = '%d-%b'
            formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y', '%d %b %Y %H:%M', ]
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            lines = []
            # naming the legend and drawing the lines
            for sim_name in res_dict.keys():

                line_label_long = ''
                for var_x in changing_vars:
                    val_x = study_df_reduced.loc[sim_name][var_x]
                    line_label_x = str(var_x.replace("_", " ")) + ' = ' + str(val_x)  # + '\n'
                    line_label_long += line_label_x

                line_label_short = study_df_reduced.at[sim_name, 'short_sim_name']

                # if there a lot of Sims, use the short line label. Otherwise the Legend is too big.
                if len(study_df_reduced) > 5:
                    line_label = line_label_short
                else:
                    line_label = line_label_long

                # resample:H: hourly, D: daily, S: seconds (f.e 600S)
                line = ax.plot(res_dict[sim_name][var][start_date:end_date].resample(resample_delta).mean(),
                               linewidth=0.7, label=line_label)
                lines += line

            ax.set_ylabel(y_label.replace("_", " "))
            # ax.set_xlabel("Time in Date")

            labs = [line.get_label() for line in lines]
            ax.legend(lines, labs, loc="best", borderaxespad=0.5, ncol=2, fontsize=10)
            # fig.autofmt_xdate(rotation=45, ha="center")

            fig_lst.append(fig)
            plt.show()

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
    long_sim_titles = True
    plt.style.use("/Users/jonasgrossmann/git_repos/matplolib-style/ebc.paper.mplstyle")
    sns.set()
    sns.set_context("paper", rc={"lines.linewidth": 1.5})  # style of the plot
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
            plt.show()

            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".pdf"))
                # fig.savefig(os.path.join(dir_output, sim_name + "_" + savename_append + ".png"), dpi=600)

    # ------------ Single Variable, one line is one simulation, this is what Dymola CANT do -------------------

    elif plot_style == 'Single Variable':
        for var in var_lst:

            # ----- Data Selection -------
            data_df = pd.DataFrame()  # stores all the lines as columns
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
            fig, ax = plt.subplots()
            fig = sns.lineplot(data=data_df, dashes=False)
            # legend = plt.legend(bbox_to_anchor=(0, 1.02, 1, 1), loc="lower left",
            #            mode="expand", borderaxespad=0, ncol=2)
            legend = plt.legend(loc='best', ncol=8, fontsize=10)
            plt.title(var)
            # plt.tight_layout()

            # date_form = mdates.DateFormatter('%b')
            # date_loc = mdates.MonthLocator()
            # ax.xaxis.set_major_formatter(date_form)
            # ax.xaxis.set_major_locator(date_loc)

            fig = plt.gcf()  # get current figure
            fig_lst.append(fig)
            plt.show()

            # ----- Save Figure ------
            if save_figs:
                if not os.path.exists(dir_output):
                    os.makedirs(dir_output)
                fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".pdf"))
                # fig.savefig(os.path.join(dir_output, var + "_" + savename_append + ".png"), dpi=600)

    return fig_lst


def compute_w_tot_df(dir_sciebo, dir_case, case):
    """
    right now only works with one .mat file at a time!

    :param dir_sciebo:
    :param case:
    :return:
    """

    dir_case = Path(dir_case)  # Pathlib Object
    study_csv = dir_case / "study.csv"
    study_df = pd.read_csv(study_csv, index_col=0)

    # make new columns for the KPIs
    kpi_W_tot_kWh = 'W_tot_kWh'
    study_df[kpi_W_tot_kWh] = 0
    kpi_W_Central_Pump_kWh = 'W_Central_Pump_kWh'
    study_df[kpi_W_Central_Pump_kWh] = 0
    kpi_w_substation_kWh = 'W_Substation_kWh'
    study_df[kpi_w_substation_kWh] = 0

    if case == 'Case1':

        kpi_W_Central_HP_kWh = 'W_Central_HP_kWh'
        study_df[kpi_W_Central_HP_kWh] = 0

        check_case(dir_models=dir_case, case=case)

        # read all variables, get desired lists of variable names
        case1_vars_lst = read_trajectory_names_folder(res_dir=dir_case, only_from_first_sim=True)
        power_central_pump = make_sublist(case1_vars_lst, 'Supply.fan.P')
        power_substation = make_sublist(case1_vars_lst, end='heaPum.P_el.y')
        power_central_hp = make_sublist(case1_vars_lst, end='Supply.rev_hea_pum.P')
        power_all = power_central_pump + power_substation + power_central_hp

        # convert mat files to dataframes, takes long!
        p_tot_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=power_all)
        p_central_pump_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=power_central_pump)
        p_central_hp_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=power_central_hp)
        p_substations_hp_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=power_substation)

        # for each sim, compute the KPIs and write them to study.csv
        for sim_name in study_df.index:
            w_tot_kWh = compute_w_tot_single_df(sim_dataframe=p_tot_dict[sim_name])
            w_central_pump_kWh = compute_w_tot_single_df(sim_dataframe=p_central_pump_dict[sim_name])
            w_central_hp_kWh = compute_w_tot_single_df(sim_dataframe=p_central_hp_dict[sim_name])
            w_substation_hp_kWh = compute_w_tot_single_df(sim_dataframe=p_substations_hp_dict[sim_name])

            # write the KPI to the study_df
            study_df.loc[[sim_name], [kpi_W_tot_kWh]] = int(w_tot_kWh)
            study_df.loc[[sim_name], [kpi_W_Central_Pump_kWh]] = int(w_central_pump_kWh)
            study_df.loc[[sim_name], [kpi_W_Central_HP_kWh]] = int(w_central_hp_kWh)
            study_df.loc[[sim_name], [kpi_w_substation_kWh]] = int(w_substation_hp_kWh)
            study_df.to_csv(dir_case / "study.csv")

    elif case == 'CaseBase':

        check_case(dir_models=dir_case, case=case)

        # read all variables, get desired variable names and their corresponding dataframes
        case1_vars_lst = read_trajectory_names_folder(res_dir=dir_case, only_from_first_sim=True)
        power_central_pump = make_sublist(case1_vars_lst, 'Supply.fan.P')
        power_central_pump_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=power_central_pump)

        # for each sim, compute the KPI and write it to study_df
        for sim_name in power_central_pump_dict.keys():
            w_central_pump_kWh = compute_w_tot_single_df(sim_dataframe=power_central_pump_dict[sim_name])

            # ----- individual chillers, not part of the simulation -----
            x, cold_demand = import_demands_from_demgen(dir_sciebo, house_type='Old', output_interval=3600)  # Wh
            total_cold_dem = sum(cold_demand) / 1000  # kWh
            cop_cc = 5
            w_tot_cc_kWh = total_cold_dem / cop_cc

            w_tot_kWh = w_central_pump_kWh + w_tot_cc_kWh

            # write the KPI to the study_df and save it as an updated version of the study.csv
            study_df.loc[[sim_name], [kpi_W_tot_kWh]] = int(w_tot_kWh)
            study_df.loc[[sim_name], [kpi_w_substation_kWh]] = int(w_tot_cc_kWh)
            study_df.loc[[sim_name], [kpi_W_Central_Pump_kWh]] = int(w_central_pump_kWh)
            study_df.to_csv(dir_case / "study.csv")

    else:
        raise Exception("wrong case")

    return study_df[kpi_W_tot_kWh]  # kWh


def compute_q_tot_df(dir_case, case):
    """
    right now olny works with one .mat file at a time!

    :param dir_case:
    :param case:
    :return:
    """

    dir_case = Path(dir_case)  # Pathlib Object
    study_csv = dir_case / "study.csv"
    study_df = pd.read_csv(study_csv, index_col=0)

    # make new column for the KPI
    kpi_name = 'Q_tot_kWh'
    study_df[kpi_name] = 0

    if case == 'Case1':

        check_case(dir_models=dir_case, case=case)

        q_tot_kWh = 0  # Case 1 is fully electric
        study_df[kpi_name] = q_tot_kWh
        study_df.to_csv(dir_case / "study.csv")

    elif case == 'CaseBase':

        check_case(dir_models=dir_case, case=case)

        # read all variables, get desired variable names, get the dataframes
        basecase_vars_lst = read_trajectory_names_folder(res_dir=dir_case, only_from_first_sim=True)
        q_flow_central_heater = make_sublist(basecase_vars_lst, end='Supply.heater.Q_flow')
        q_flow_central_heater_dict = mat_files_filtered_to_dict(res_dir=dir_case, var_lst=q_flow_central_heater)

        # for each sim, compute the KPI and write it to study_df
        for sim_name in q_flow_central_heater_dict.keys():
            q_central_heater_kWh = compute_w_tot_single_df(sim_dataframe=q_flow_central_heater_dict[sim_name])
            q_tot_kWh = q_central_heater_kWh

            study_df.loc[[sim_name], [kpi_name]] = q_tot_kWh
            study_df.to_csv(dir_case / "study.csv")

    else:
        raise Exception("wrong case")

    return study_df[kpi_name]  # kWh


def write_const_to_study_csv(dir_models, const_name_end, column_name):
    """
    write a constant to the study df. Example:  column_name: 'hyraulic diameter'
                                                const_name_end: 'cor.del.dh'
    :param dir_models:
    :param const_name_end:
    :param column_name:
    :return: study_df:
    """

    dir_models = Path(dir_models)  # Pathlib Object
    study_df = pd.read_csv(dir_models / "study.csv", index_col=0)

    # make new column for the variable
    study_df[column_name] = 0

    vars_lst = read_constants_names_folder(res_dir=dir_models, only_from_first_sim=True)
    const_lst = make_sublist(vars_lst, end=const_name_end)
    const_lst_dict = mat_files_filtered_to_dict(res_dir=dir_models, var_lst=const_lst)

    # for each sim, compute the KPI and write it to study_df
    for sim_name in const_lst_dict.keys():
        df = const_lst_dict[sim_name]

        var = df.loc[['2019-12-31 23:10:00'], []]  # first entry

        study_df.loc[[sim_name], [column_name]] = var
        study_df.to_csv(dir_models / "study.csv")

    return study_df


def compute_gwi_df(dir_case, case):
    """
    Computes the Global Warming Impact (GWI) in [kg_CO2]

    :param dir_case:
    :param case:
    :return:
    """

    dir_case = Path(dir_case)  # Pathlib Object
    study_csv = dir_case / "study.csv"
    study_df = pd.read_csv(study_csv, index_col=0)

    if case == 'Case1':

        check_case(dir_models=dir_case, case=case)

        kpi_gwi_tot = 'GWI_tot'
        study_df[kpi_gwi_tot] = 0

        # for each sim, compute the KPI and write it to study df reduced -> better normal study.csv?
        for sim_name in study_df.index:
            # get the total Electricity from the study.csv
            w_tot_kWh = study_df.at[sim_name, "W_tot_kWh"]

            k_co2 = 0.468  # kg co2 per kWh Strom https://www.umweltbundesamt.de/presse/pressemitteilungen/bilanz-2019-co2-emissionen-pro-kilowattstunde-strom
            gwi = w_tot_kWh * k_co2

            study_df.loc[[sim_name], [kpi_gwi_tot]] = int(gwi)
            study_df.to_csv(dir_case / "study.csv")

    elif case == 'CaseBase':

        check_case(dir_models=dir_case, case=case)

        kpi_gwi_hkw = 'GWI_HKW'
        study_df[kpi_gwi_hkw] = 0
        kpi_gwi_bhkw = 'GWI_BHKW'
        study_df[kpi_gwi_bhkw] = 0
        kpi_gwi_kkw = 'GWI_KKW'
        study_df[kpi_gwi_kkw] = 0

        kpi_gwi_tot_hkw = 'GWI_tot_HKW'
        study_df[kpi_gwi_tot_hkw] = 0
        kpi_gwi_tot_bhkw = 'GWI_tot_BHKW'
        study_df[kpi_gwi_tot_bhkw] = 0
        kpi_gwi_tot_kkw = 'GWI_tot_KKW'
        study_df[kpi_gwi_tot_kkw] = 0

        kpi_gwi_central_pump = 'GWI_Central_Pump'
        study_df[kpi_gwi_central_pump] = 0

        f_p_hkw = 1.3  # Fernwärme mit Heizwerken, keine gleichzeitige Produktion von Strom, Source: DIN V 18599-1
        f_p_bhkw = 0.7  # Fernwärme mit KWK (Gud/CHP), Strom und Wärme, Source: DIN V 18599-1
        f_p_kkw = 1.1  # ????? Todo: find source!

        x_co2_coal = 0.43  # kg co2 per kWh Braunkohle, Source: DIN V 18599-1
        x_co2_gas = 0.24  # kg co2 per kWh Erdgas, Source: DIN V 18599-1

        m_coal = 16 / 100  # Werte aus Vattenfall Berlin
        m_gas = 1 - m_coal
        x_co2_mix = x_co2_coal * m_coal + x_co2_gas * m_gas

        x_co2_elec = 0.468  # kg co2 per kWh Strom https://www.umweltbundesamt.de/presse/pressemitteilungen/bilanz-2019-co2-emissionen-pro-kilowattstunde-strom

        # for each sim, compute the KPIs and write them to study_df
        for sim_name in study_df.index:
            q_central_heater_kWh = study_df.at[sim_name, "Q_tot_kWh"]

            gwi_central_heater_hkw_gas = q_central_heater_kWh * f_p_hkw * x_co2_gas
            gwi_central_heater_bhkw_gas = q_central_heater_kWh * f_p_bhkw * x_co2_gas
            gwi_central_heater_kkw_coal = q_central_heater_kWh * f_p_kkw * x_co2_coal

            w_tot_kWh = study_df.at[sim_name, "W_tot_kWh"]
            gwi_elec = w_tot_kWh * x_co2_elec

            # total GWI = Heater + Electricity
            gwi_tot_hkw_gas = gwi_central_heater_hkw_gas + gwi_elec
            gwi_tot_bhkw_gas = gwi_central_heater_bhkw_gas + gwi_elec
            gwi_tot_kkw_coal = gwi_central_heater_kkw_coal + gwi_elec

            # write all KPIs to the study_df
            study_df.loc[[sim_name], [kpi_gwi_central_pump]] = int(gwi_elec)

            study_df.loc[[sim_name], [kpi_gwi_hkw]] = int(gwi_central_heater_hkw_gas)
            study_df.loc[[sim_name], [kpi_gwi_bhkw]] = int(gwi_central_heater_bhkw_gas)
            study_df.loc[[sim_name], [kpi_gwi_kkw]] = int(gwi_central_heater_kkw_coal)

            study_df.loc[[sim_name], [kpi_gwi_tot_hkw]] = int(gwi_tot_hkw_gas)
            study_df.loc[[sim_name], [kpi_gwi_tot_bhkw]] = int(gwi_tot_bhkw_gas)
            study_df.loc[[sim_name], [kpi_gwi_tot_kkw]] = int(gwi_tot_kkw_coal)

            study_df.to_csv(dir_case / "study.csv")

    else:
        raise Exception("wrong case")

    return study_df


def compute_op_df(dir_case, case):
    """
    right now only works with one .mat file at a time!

    :param dir_case:
    :param case:
    :return:
    """

    dir_case = Path(dir_case)  # Pathlib Object
    study_csv = dir_case / "study.csv"
    study_df = pd.read_csv(study_csv, index_col=0)

    if case == 'Case1':

        check_case(dir_models=dir_case, case=case)

        kpi_name = 'OP'
        study_df[kpi_name] = 0

        # for each sim, compute the KPI and write it to study.csv
        for sim_name in study_df.index:
            # get the total Electricity from the study.csv
            w_tot_kWh = study_df.at[sim_name, "W_tot_kWh"]
            price_w_tot = 0.25  # € per kWh Strom
            op_w_tot = w_tot_kWh * price_w_tot

            # get the total Electricity from the study.csv
            gwi = study_df.at[sim_name, "GWI_tot"]
            price_gwi = 0.03  # 30€ per Ton -> 0,3€ per kg CO2
            op_gwi = gwi * price_gwi

            op = op_gwi + op_w_tot

            # write the KPI to the study_df
            study_df.loc[[sim_name], [kpi_name]] = int(op)
            study_df.to_csv(dir_case / "study.csv")

    elif case == 'CaseBase':

        check_case(dir_models=dir_case, case=case)

        kpi_OP_tot_HKW = 'OP_tot_HKW'
        study_df[kpi_OP_tot_HKW] = 0
        kpi_OP_tot_BHKW = 'OP_tot_BHKW'
        study_df[kpi_OP_tot_BHKW] = 0
        kpi_OP_tot_KKW = 'OP_tot_KKW'
        study_df[kpi_OP_tot_KKW] = 0

        f_p_hkw = 1.3  # Fernwärme mit Heizwerken, keine gleichzeitige Produktion von Strom, Source: DIN V 18599-1
        f_p_bhkw = 0.7  # Fernwärme mit KWK (Gud/CHP), Strom und Wärme, Source: DIN V 18599-1
        f_p_kkw = 1.3  #

        c_gas = 0.02  # Gasprice in € per kWh Gas, Source: BDEW https://www.bdew.de/media/documents/201013_BDEW-Gaspreisanalyse_Juli_2020.pdf

        kWh_per_kg_coal = 4.17  # https://www.agrarplus.at/heizwerte-aequivalente.html
        price_kg_coal = 0.05  # € per kg coal https://markets.businessinsider.com/commodities/coal-price?op=1
        kWh_per_kg_coal = 7.97  # Source: https://books.google.de/books?id=n0fVYjrHAlwC&lpg=PA58&pg=PP1#v=onepage&q&f=false
        price_kg_coal = 0.095  # € per kg steinkohle https://www.bafa.de/DE/Energie/Rohstoffe/Drittlandskohlepreis/drittlandskohlepreis_node.html

        c_coal = price_kg_coal / kWh_per_kg_coal  # Coalprice in € per kWh Gas

        # for each sim, compute the KPI and write it to study.csv
        for sim_name in study_df.index:
            # get total Heat from the study.csv
            q_central_heater_kWh = study_df.at[sim_name, "Q_tot_kWh"]

            op_central_heater_hkw_gas = q_central_heater_kWh * f_p_hkw * c_gas
            op_central_heater_bhkw_gas = q_central_heater_kWh * f_p_bhkw * c_gas
            op_central_heater_kkw_coal = q_central_heater_kWh * f_p_kkw * c_coal

            # get the total Electricity from the study.csv
            w_tot_kWh = study_df.at[sim_name, "W_tot_kWh"]
            price_w_tot = 0.25  # € per kWh Strom
            op_w_tot = w_tot_kWh * price_w_tot

            # get the GWIs from the study.csv
            gwi_tot_hkw = study_df.at[sim_name, "GWI_tot_HKW"]
            gwi_tot_bhkw = study_df.at[sim_name, "GWI_tot_BHKW"]
            gwi_tot_kkw = study_df.at[sim_name, "GWI_tot_KKW"]

            price_gwi = 0.03  # 30€ per Ton -> 0,3€ per kg CO2
            op_gwi_hkw = gwi_tot_hkw * price_gwi
            op_gwi_bhkw = gwi_tot_bhkw * price_gwi
            op_gwi_kkw = gwi_tot_kkw * price_gwi

            # Total Operational Costs
            op_tot_hkw = op_central_heater_hkw_gas + op_gwi_hkw + op_w_tot
            op_tot_bhkw = op_central_heater_bhkw_gas + op_gwi_bhkw + op_w_tot
            op_tot_kkw = op_central_heater_kkw_coal + op_gwi_kkw + op_w_tot

            # write the KPI to the study_df
            study_df.loc[[sim_name], [kpi_OP_tot_HKW]] = int(op_tot_hkw)
            study_df.loc[[sim_name], [kpi_OP_tot_BHKW]] = int(op_tot_bhkw)
            study_df.loc[[sim_name], [kpi_OP_tot_KKW]] = int(op_tot_kkw)
            study_df.to_csv(dir_case / "study.csv")

    else:
        raise Exception("wrong case")

    return study_df  # kWh


def compute_all_KPIs(dir_sciebo, dir_models, case):
    """
    computes all the KPIs of one Case, then reduces the updated study.csv and returns it.
    :param dir_sciebo:              sciebo-folder, only for calling the DemGen Cold Method
    :param dir_models:              directory where the models are stored
    :param case:                    type of models, can be "CaseBase" or "Case1"
    :return: changing_vars_case,
             study_df_reduced_case
    """
    compute_w_tot_df(dir_sciebo=dir_sciebo, dir_case=dir_models, case=case)
    compute_q_tot_df(dir_case=dir_models, case=case)
    compute_gwi_df(dir_case=dir_models, case=case)
    compute_op_df(dir_case=dir_models, case=case)
    changing_vars_case, study_df_reduced_case = reduce_and_update_study_df(dir_models)

    return changing_vars_case, study_df_reduced_case


def check_case(dir_models, case):
    """
    check if the Prefix of the Folders in the directory 'dir_models' corresponds
    to the String given in 'case'. F.e. when a BaseCase Model should be analyzed as a Case1 Model.
    :param dir_models:    PathLib Directory:  Directory where Models are stored
    :param case:          String:             String that should be included in model name
    :return:
    """
    wrong_sims = [f.name for f in dir_models.iterdir() if case not in f.name and f.is_dir()]
    if len(wrong_sims) > 0:
        error_message = "A Model not from the case '{}' was found in {}".format(case, dir_models)
        print(error_message)

    return len(wrong_sims)


def compute_w_tot_single_df(sim_dataframe):
    """
    computes the total electricity of a dataframe that hold values for the power.
    W = P * dt
    analogous, computes the total heat of a dataframe that hold values for the heat flow.
    Q = Q_flow * dt

    :param sim_dataframe:
    :return:
    """

    timedelta = sim_dataframe.index[1] - sim_dataframe.index[0]
    timedelta_sec = timedelta.seconds  # timestep width of the data
    w_vars = [p_step * timedelta_sec for p_step in sim_dataframe.sum()]  # in Ws
    w_vars_kWh = [w_var / (3600 * 1000) for w_var in w_vars]  # in kWh
    w_tot_kWh = sum(w_vars_kWh)

    return w_tot_kWh


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

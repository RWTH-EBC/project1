import pandas as pd
import os
import numpy as np
import copy
from uesgraphs.uesgraph import UESGraph
import matplotlib.pyplot as plt
import matplotlib

from uesgraphs.visuals import Visuals

import networkx as nx
import matplotlib
from matplotlib.pylab import mpl
from matplotlib.collections import LineCollection
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import shapely.geometry as sg
import sys
import warnings

import streamlit as st
import seaborn as sns

from modelicares import SimRes


def main():
    res_path = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/model/"
    "Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50/"
    "Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50_inputs.mat"

    # print(len(sim.names()))
    # print(type(sim.names()))
    # print(sim.names()[0:100])

    supply_TReturn_name = sim.names()[28]
    supply_TReturn = sim[supply_TReturn_name]
    print(supply_TReturn.description)
    print(supply_TReturn.is_constant)
    print(supply_TReturn.unit)



    fan = [i for i in substation_res if i.endswith("pumpHeating.P")]  # no Substation ending with pumpHeating.P
    # fan is signal of read all files
    print(fan)

    res = read_all_files(res_path, fan)
    res     # for streamlit magic?

    # results_all = import_simulation_results_modelicares_to_df(res_path, substation_res)


@st.cache(persist=True)
def import_simulation_results_modelicares_to_df(res_path, signals):
    """
    Imports results from network simulation from a .mat file

    Parameters
    ----------
    res_path : str
        Path to result file
    signals : list
        list of variables to import from result file

    Returns
    --------
    res_all : pandas data frame including the given varaibles of the given result file
    """

    sim = SimRes(res_path)

    res_all = sim.to_pandas(signals, with_unit=False)

    res_all = res_all.groupby(res_all.index).first()
    res_all = res_all[res_all.index.isin(range(0, 31536000, 900))]

    res_all.index = res_all.index.astype(int)
    res_all.index = pd.to_datetime(res_all.index, unit="s", origin="2019")

    return res_all


@st.cache(persist=True)
def read_all_files(res_path, signals):
    """
    Imports results with 'import_simulation_results_modelicares_to_df' function. Then converts the dataframe to a csv
    :param res_path: str
        Path to the result file
    :param signals: list
        list of variables to import from result file
    :return res_dict: dictionary
        dictionary necessary for streamlit magic stuff?
    """

    res_dict = {}
    res_all = import_simulation_results_modelicares_to_df(res_path, signals)

    # takes last string of result file, cuts of the last 4 elements ('.mat' ending) and adds another string to the end
    res_all.to_csv(res_path.split("/")[-1][:-4] + "_pumping_power_modelicares.csv")
    res_dict[res_path] = res_all

    return res_dict


def read_variable_names_filtered(res_path):
    '''

    :param res_path:
    :return:
    '''

    sim = SimRes(fname=res_path)
    all_vars_lst = sim.names()

    # add all pipe variables from the sim_vars_list into a new list
    pipe_res = [i for i in all_vars_lst if "Pipe" in i]
    print("There are " + str(len(pipe_res)) + " Pipe variables in the results file")

    # add all demand variables from the sim_vars_list into a new list
    substation_res = [i for i in all_vars_lst if "demand" in i]
    print("There are " + str(len(substation_res)) + " Substation variables in the results file")

    # add all supply variables from the sim_vars_list into a new list
    supply_res = [i for i in all_vars_lst if "networkModel.supply" in i]
    print("There are " + str(len(supply_res)) + " Supply variables in the results file")

    return pipe_res, substation_res, supply_res

if __name__ == '__main__':
    main()

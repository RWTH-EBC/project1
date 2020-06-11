# -*- coding: utf-8 -*-
# @Author: MichaMans
# @Date:   2020-02-14 14:55:15
# @Last Modified by:   MichaMans
# @Last Modified time: 2020-05-13 18:12:23

# from dymola.dymola_interface import DymolaInterface
import pandas as pd
import os
import numpy as np
import copy
from uesgraphs.uesgraph import UESGraph
import matplotlib

from uesgraphs.visuals import Visuals

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
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

# plt.style.use('~/ebc.paper.mplstyle')

sns.set()
sns.set_context("paper")


def main():
    """
    saves path and directory where .mat file is stored.

    function calls:
        - 'read_trajectory_names_filtered'
        - 'read_all_files'
        - 'analyse_pumping_power_supply_decentral'
    """

    st.title("Erdeis II Plot Configuration")

    res_dir = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/mat_files_from_models"

    res_path = "/Users/jonasgrossmann/git_repos/project1/wp_3_2_destest/Network/NetworkSizing/model" \
               "/Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50" \
               "/Destest_Jonas__T_82_60_62__dT_20__p_6_3_2__mBy_50_inputs.mat"

    dir_this = os.path.abspath(os.path.dirname(__file__))
    dir_src = os.path.abspath(os.path.dirname(dir_this))
    # dir_top = os.path.abspath(os.path.dirname(dir_src))
    dir_workspace = os.path.abspath(os.path.join(dir_src, "workspace"))
    dir_output = os.path.abspath(os.path.join(dir_workspace, "plots"))

    pipes, substations, supply = read_trajectory_names_filtered(res_path)

    fan = [i for i in substations if i.endswith("pumpHeating.P")]

    res = read_all_files(res_dir, fan)
    res #streamlit magic command?

    for key in res.keys():
        analyse_pumping_power_supply_decentral(res[key], key, dir_output)


def analyse_pumping_power_supply_decentral(res, key, dir_output):

    # res.to_csv("Sim20200130154813wichelkoppelnZentralschleswigerdiameterminus1_pumping_power.csv")

    # "data", res.keys()

    # res = res["D:\\dymola\\Erdeis\\Zentral\\Sim20200130154755wichelkoppelnZentralselfhydronicsizing.mat"].resample("H").mean()

    # ax1 = res_h["networkModel.supplySupply_0.fan.P"].plot()

    name = key.split("/")[-1][:-4]

    fig1, ax1 = plt.subplots()

    fig1.suptitle(name, fontsize=5)

    res_p = res.filter(like=".pumpHeating.P").sum(axis=1).resample("H").mean()

    print(res_p)

    ax1.plot(
        res_p.resample("D").mean() / 1000,
        label="Total pump power: %s kWh" % (
            round(res_p.sum() / 1000, 2)))

    ax1.set_ylabel(r"Pumping Power in kW")
    ax1.set_xlabel(r"Time in Date")

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0.)

    fig1.autofmt_xdate(rotation=45, ha="center")

    fig1.savefig(os.path.join(dir_output, name + "_pumping_power.pdf"))
    fig1.savefig(os.path.join(dir_output, name + "_pumping_power.png"), dpi=600)

    st.pyplot(fig1)

    # ax1 = res_h["networkModel.supplySupply_0.fan.P"].plot()

    fig2, ax2 = plt.subplots()

    fig2.suptitle(name, fontsize=5)

    l1 = ax2.plot(
        res["networkModel.supplySupply_0.port_b.p"].resample("D").mean() / 100000 -
        res["networkModel.supplySupply_0.port_a.p"].resample("D").mean() / 100000,
        label=r"Pressure at supply: $\varnothing$ %s Bar" % (
            round(res["networkModel.supplySupply_0.port_a.p"].sum() / 100000 / 8760, 2)),
        color="m")

    ax2.set_ylabel(r"Pressure Head in bar")

    ax2.set_ylim(4.5, 6.5)
    ax2.set_xlabel(r"Time in Date")

    ax3 = ax2.twinx()
    l2 = ax3.plot(
        res["networkModel.supplySupply_0.port_b.m_flow"].resample("D").min() * -1,
        label="Massflow rate" % (
            round(res["networkModel.supplySupply_0.port_b.p"].sum() / 100000 / 8760, 2)),
        color="g")

    ax3.set_ylabel(r"Massflow in kg/s")
    ax3.set_ylim(2, 18)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]

    ax2.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0., ncol=2)

    fig2.autofmt_xdate(rotation=45, ha="center")

    fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.pdf"))
    fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.png"), dpi=600)

    st.pyplot(fig2)

    return res


def analyse_pumping_power_supply_central(res, key, dir_output):

    # res.to_csv("Sim20200130154813wichelkoppelnZentralschleswigerdiameterminus1_pumping_power.csv")

    # "data", res.keys()

    # res = res["D:\\dymola\\Erdeis\\Zentral\\Sim20200130154755wichelkoppelnZentralselfhydronicsizing.mat"].resample("H").mean()

    # ax1 = res_h["networkModel.supplySupply_0.fan.P"].plot()

    name = key.split("/")[-1][:-4]

    fig1, ax1 = plt.subplots()

    fig1.suptitle(name, fontsize=5)

    ax1.plot(
        res["networkModel.supplySupply_0.fan.P"].resample("D").mean() / 1000,
        label="Total pump power: %s kWh" % (
            round(res["networkModel.supplySupply_0.fan.P"].sum() / 1000, 2)))

    ax1.set_ylabel(r"Pumping Power in kW")
    ax1.set_xlabel(r"Time in Date")

    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0.)

    fig1.autofmt_xdate(rotation=45, ha="center")

    fig1.savefig(os.path.join(dir_output, name + "_pumping_power.pdf"))
    fig1.savefig(os.path.join(dir_output, name + "_pumping_power.png"), dpi=600)

    st.pyplot(fig1)

    # ax1 = res_h["networkModel.supplySupply_0.fan.P"].plot()

    fig2, ax2 = plt.subplots()

    fig2.suptitle(name, fontsize=5)

    l1 = ax2.plot(
        res["networkModel.supplySupply_0.port_b.p"].resample("D").mean() / 100000 -
        res["networkModel.supplySupply_0.port_a.p"].resample("D").mean() / 100000,
        label=r"Pressure at supply: $\varnothing$ %s Bar" % (
            round(res["networkModel.supplySupply_0.port_a.p"].sum() / 100000 / 8760, 2)),
        color="m")

    ax2.set_ylabel(r"Pressure Head in bar")

    ax2.set_ylim(4.5, 6.5)
    ax2.set_xlabel(r"Time in Date")

    ax3 = ax2.twinx()
    l2 = ax3.plot(
        res["networkModel.supplySupply_0.port_b.m_flow"].resample("D").min() * -1,
        label="Massflow rate" % (
            round(res["networkModel.supplySupply_0.port_b.p"].sum() / 100000 / 8760, 2)),
        color="g")

    ax3.set_ylabel(r"Massflow in kg/s")
    ax3.set_ylim(2, 18)

    lns = l1 + l2
    labs = [l.get_label() for l in lns]

    ax2.legend(lns, labs, bbox_to_anchor=(0., 1.02, 1., .102), loc="lower center", borderaxespad=0., ncol=2)

    fig2.autofmt_xdate(rotation=45, ha="center")

    fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.pdf"))
    fig2.savefig(os.path.join(dir_output, name + "_massflow_pressure.png"), dpi=600)

    st.pyplot(fig2)

    return res


@st.cache(persist=True)
def read_all_files(res_dir, signals):

    files = [os.path.join(res_dir, f) for f in os.listdir(res_dir) if (os.path.isfile(os.path.join(res_dir, f)) and f.endswith(".mat"))]
    print("files", files)
    res_dict = {}

    for fil in files:
        res_all = import_simulation_results_modelicares(fil, signals)
        res_all.to_csv(fil.split("\\")[-1][:-4] + "_pumping_power_modelicares.csv")
        res_dict[fil] = res_all
    return res_dict


@st.cache(persist=True)
def import_simulation_results(res_path, signals):
    """
    Imports results from network simulation from a .mat file

    Parameters
    ----------
    res_path : str
        Path to result file
    """

    res_all = pd.DataFrame()

    if len(signals) > 25:
        chunks = [signals[i : i + 25] for i in range(0, len(signals), 25)]
    else:
        chunks = signals

    for index, chunk in enumerate(chunks):
        dymola = DymolaInterface()
        print("Reading chunk number ", index + 1, " out of total: ", len(chunks))
        print("Reading", *chunk, " from ", res_path, sep="\n")
        chunk = chunk + ["Time"]

        dym_res = dymola.readTrajectory(
            fileName=res_path,
            signals=chunk,
            rows=dymola.readTrajectorySize(fileName=res_path),
        )
        results = pd.DataFrame().from_records(dym_res).T
        results = results.rename(columns=dict(zip(results.columns.values, chunk)))
        results.index = results["Time"]
        results = results.rename(columns={"Time": "Time_Step"})

        # I leave this here for non equidistant timesteps

        # x = range(0, 31536000, 900)
        # index_to_drop = []
        # j = 1
        # print('starting to iterate over df')
        # for i, row in enumerate(results.iterrows()):
        #     if results.index[i] not in x:
        #         index_to_drop.append(results.index[i])

        # results = results.drop(index_to_drop)
        results = results.groupby(level=0).first()

        # results.to_csv(path=res_path, delimiter=';')
        dymola.close()

        if res_all.empty:
            res_all = results
        else:
            res_all = res_all.merge(results)

    res_all = res_all.rename(columns={"Time_Step": "Time"})
    res_all.index = res_all["Time"]

    res_all = res_all.groupby(res_all.index).first()
    res_all = res_all[res_all.index.isin(range(0, 31536000, 900))]

    res_all.index = res_all.index.astype(int)
    res_all.index = pd.to_datetime(res_all.index, unit="s", origin="2019")

    return res_all


@st.cache(persist=True)
def import_simulation_results_modelicares(res_path, signals):
    """
    Imports results from network simulation from a .mat file

    Parameters
    ----------
    res_path : str
        Path to result file
    """

    res_all = pd.DataFrame()

    sim = SimRes(res_path)

    results = sim.to_pandas(signals, with_unit=False)

    res_all = results

    res_all = res_all.groupby(res_all.index).first()
    res_all = res_all[res_all.index.isin(range(0, 31536000, 900))]

    res_all.index = res_all.index.astype(int)
    res_all.index = pd.to_datetime(res_all.index, unit="s", origin="2019")

    return res_all


def read_trajectory_names_filtered(res_path):
    """
    Imports results from network simulation from a .mat file
@st.cache(persist=True)
    Parameters
    ----------
    res_path : str
        Path to result file
    """

    print("Reading trajectory names of ", res_path)

    sim = SimRes(fname=res_path)

    dym_res = sim.get_trajectories()

    # dym_res = dymola.readTrajectoryNames(fileName=res_path)

    # pipes
    pipe_res = [i for i in dym_res if "Pipe" in i]

    # substations
    substation_res = [i for i in dym_res if "demand" in i]

    # supply
    supply_res = [i for i in dym_res if "networkModel.supply" in i]

    return pipe_res, substation_res, supply_res


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import numpy as np
import os
import platform
from datetime import datetime
from uesgraphs.systemmodels import utilities as sysmod_utils
from uesgraphs.systemmodels import systemmodelheating as sysmh
from uesgraphs.uesmodels.utilities import utilities as utils
import itertools
from sklearn.model_selection import ParameterGrid
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import fnmatch
import shutil

import pycity_base.classes.demand.domestic_hot_water as dhw
import pycity_base.classes.timer as time
import pycity_base.classes.weather as weath
import pycity_base.classes.prices as price
import pycity_base.classes.environment as env
import pycity_base.classes.demand.occupancy as occ


# import csv


def main():
    # --------------------------- paths ---------------------------
    if platform.system() == 'Darwin':
        dir_sciebo = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    elif platform.system() == 'Windows':
        dir_sciebo = "D:/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    else:
        raise Exception("Unknown operating system")

    # ------------------------------------ demand profiles---------------------------------------------
    ground_temps = import_ground_temp_table(dir_sciebo, plot_series=False, start_in_summer=True)

    # dhw_demand_pycity = generate_dhw_profile_pycity()
    dhw_demand = import_from_dhwcalc(dir_sciebo, plot_demand=False, start_in_summer=True)
    # heat_demand_ibpsa = import_demands_from_github()
    heat_demand_old, cold_demand_old = import_demands_from_demgen(dir_sciebo, house_type='Old',plot_demand=False,
                                                                  start_in_summer=False)
    heat_demand_standard, cold_demand_standard = import_demands_from_demgen(dir_sciebo, house_type='Standard',
                                                                            plot_demand=False, start_in_summer=False)

    heat_and_dhw_demand_basecase = [sum(i) for i in zip(heat_demand_old, dhw_demand)]

    storage_load = convert_dhw_load_to_storage_load(dhw_demand, plot_demand=True)


    # ------------------------ Parameter Dictionaries to pass into parameter study function -------------------------
    aixlib_dhc = "AixLib.Fluid.DistrictHeatingCooling."

    params_dict_base_case = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 80],
        'graph__p_nominal': [3e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': ground_temps,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': 'AixLib.Fluid.FixedResistances.PlugFlowPipe',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,
        'edge__kIns': [0.035],
        # 'edge__kIns': [0.002, 0.035],
        'edge__dIns': [0.045],
        # 'edge__dIns': [0.0025, 0.045],
        'pipe_scaling_factor': [1],
        # ------------------------------------ Demand/House Node Data ---------------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlled_BaseCase",
        # 'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpDirectCoolingDHWnoStoragePIcontrol",
        'demand__Q_flow_input': heat_and_dhw_demand_basecase,
        'demand__max_demand_heating': max(heat_and_dhw_demand_basecase),  # for size_hydraulic_network
        'input_heat_str': 'Q_flow_input',
        # 'demand__cold_input': cold_demand,
        # 'demand__dhw_input': storage_load,
        # 'demand__T_dhw_supply': [273.15 + 60],  # T_VL DHW
        'demand__dT_Network': [20],
        'demand__Q_flow_nominal': max(heat_and_dhw_demand_basecase),
        # 'demand__val_pos_min': [0.2],
        # 'demand__m_flow_min': [0.08],
        # 'demand__heatDemand_max': max_heat_demand,
        # 'demand__coldDemand_max': max_cold_demand,

        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
        'supply__TIn': [273.15 + 90],  # -> t_supply
        'supply__dpIn': [5e5],  # p_supply
        # ------------------------------------ Pressure Control ------------------------------------------
        "pressure_control_supply": "Destest_Supply",  # Name of the supply that controls the pressure
        "pressure_control_dp": [0.5e5],  # Pressure difference to be held at reference building
        "pressure_control_building": "max_distance",  # reference building for the network
        "pressure_control_p_max": [4e5],  # Maximum pressure allowed for the pressure controller
        "pressure_control_k": 12,  # gain of controller
        "pressure_control_ti": 5,  # time constant for integrator block
    }

    params_dict_case1b = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 11.14],  # Start value for every pipe? Water Properties?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': ground_temps,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': 'AixLib.Fluid.FixedResistances.PlugFlowPipe',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,
        'edge__kIns': [0.035],
        'edge__dIns': [0.045],
        'pipe_scaling_factor': [1],
        'size_dp_set': 200.0,
        # ------------------------------------ Demand/House Node Data ---------------------------------
        # 'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpDirectCoolingDHWnoStorageV2",
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpDirectCoolingDHWnoStoragePIcontrol",
        'demand__heat_input': heat_demand_old,
        'demand__max_demand_heating': max(heat_demand_old),  # for size_hydraulic_network
        'input_heat_str': 'heat_input',
        'demand__cold_input': cold_demand_standard,
        'demand__dhw_input': storage_load,
        'demand__T_dhw_supply': [273.15 + 60],  # T_VL DHW
        'demand__dT_Network': [5],
        # 'demand__val_pos_min': [0.2],
        'demand__m_flow_min': [0.08],
        'demand__heatDemand_max': max(heat_demand_old),
        'demand__coldDemand_max': max(cold_demand_standard),

        # ------------------------------ Supply Node Data --------------------------
        # 'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPumpHPandCC',
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPumpRevHP',
        'supply__TIn': [273.15 + 20],  # -> t_supply
        'supply__dpIn': [5e5],  # p_supply
        "supply__TIn_HP_Source": 273.15 + 20,
        "supply__NetworkcoldDemand_max": 16 * max(cold_demand_standard),
        "supply__NetworkheatDemand_max": 16 * max(heat_demand_old),
        # ------------------------------------ Pressure Control ------------------------------------------
        "pressure_control_supply": "Destest_Supply",  # Name of the supply that controls the pressure
        "pressure_control_dp": [0.5e5],  # Pressure difference to be held at reference building
        "pressure_control_building": "max_distance",  # reference building for the network
        "pressure_control_p_max": [4e5],  # Maximum pressure allowed for the pressure controller
        "pressure_control_k": 12,  # gain of controller
        "pressure_control_ti": 5,  # time constant for integrator block
    }

    # ---------------------------------------- create Simulations ----------------------------------------------
    parameter_study(params_dict_case1b, dir_sciebo)


def convert_dhw_load_to_storage_load(dhw_demand, s_step=600, plot_demand=False):
    """
    Converts the input DHW-Profile without a DHW-Storage to a DHW-Profile with a DHW-Storage.
    The output profile looks as if the HP would not supply the DHW-load directly but would rather re-heat
    the DHW-Storage, which has dropped below a certain dT Threshold.
    The advantage is, that no storage model has to be part of a dynamic simulation,
    although the heatpump still acts as if a storage is supplied. Based on DIN EN 12831-3.

    :param dhw_demand:      List, stores the DHW-demand profile in [W] per Timestep
    :param s_step:          Seconds within a timestep. Usual Values are 3600 (1h timesteps) or 600 (10min timesteps)
    :param plot_demand:     Plot the "Summenliniendiagram" as described in DIN DIN EN 12831-3
    :return: storage_load:  DHW-profile that re-heats a storage.
    """

    # convert the DHW demand from Watt to Joule by multiplying by the timestep width
    dhw_demand = [dem_step * s_step for dem_step in dhw_demand]

    # --------- Storage Data ---------------
    # Todo: get parameters from DIN
    # Todo: think about how Parameters should be for Schichtspeicher
    V_stor = 200   # Storage Volume in Liters
    rho = 1         # Liters to Kilograms
    m_w = V_stor * rho  # Mass Water in Storage
    c_p = 4180  # Heat Capacity Water in [J/kgK]
    dT = 55     # dT of Storage, e.f 10°C to 65°C -> 55K
    Q_full = m_w * c_p * dT

    dT_threshhold = 10   # max Temp Drop [K] in Storage before Re-heating starts
    dQ_threshhold = m_w * c_p * dT_threshhold
    Qcon_flow_max = 5000   # Heat flow rate in [W] of the HP in Storage-Heating Mode

    t_dQ = dQ_threshhold/Qcon_flow_max  # time needed to increase storage Temp by dT_threshhold in seconds [s]
    t_dQ_steps = t_dQ/s_step  # convert from seconds to timestep width

    timesteps = round(t_dQ_steps-0.5)  # -> round number of timesteps to lower integer value

    Q_dh_timesteps = dQ_threshhold * (timesteps/t_dQ_steps)     # energy added to Storage in t_dQ_steps
    Q_dh_timesteps2 = Qcon_flow_max * timesteps * s_step      # just another way to compute it
    Q_dh_timestep = Qcon_flow_max * s_step    # energy added in 1 timestep

    # ---------- write storage load time series --------
    dQ_track = 0    # tracks the cumulative dhw demand until the Temp falls more than 5K.
    storage_load = []   # new time series
    above_dT_threshhold = True

    for t_step, dem_step in enumerate(dhw_demand, start=0):

        # for initial condition, when storage_load is still empty
        if len(storage_load) == 0:
            dQ_track = dQ_track + dem_step
        else:
            dQ_track = dQ_track - storage_load[t_step - 1] + dem_step

        if above_dT_threshhold:
            if dQ_track <= Q_dh_timesteps:
                storage_load.append(0)
            else:
                above_dT_threshhold = False
                if dQ_track >= Q_dh_timestep:
                    storage_load.append(Q_dh_timestep)
                else:
                    storage_load.append(0)
            continue
        else:
            if dQ_track >= Q_dh_timestep:
                storage_load.append(Q_dh_timestep)
            else:
                storage_load.append(0)
                above_dT_threshhold = True

    # print out total demands and Difference between them
    print("Total DHW Demand is {:.2f} kWh".format(sum(dhw_demand)/(3600*1000)))
    print("Total Storage Demand is {:.2f} kWh".format(sum(storage_load) / (3600 * 1000)))
    diff = sum(dhw_demand) - sum(storage_load)
    print("Difference between dhw demand and storage load ist {:.2f} kWh".format(diff/(3600*1000)))
    if diff < 0:
        raise Exception("More heat than dhw demand is added to the storage!")

    # Count number of clusters of non-zero values ("peaks") -> reduces the amount of HP mode switches!
    dhw_peaks = int(np.diff(np.concatenate([[0], dhw_demand, [0]]) == 0).sum() / 2)
    stor_peaks = int(np.diff(np.concatenate([[0], storage_load, [0]]) == 0).sum() / 2)
    print("The Storage reduced the number of DHW heating periods from {} to {}, which is equal to "
          "{:.2f} and {:.2f} per day, respectively.".format(dhw_peaks, stor_peaks, dhw_peaks/365, stor_peaks/365))

    # Summenlinien
    dhw_demand_sumline = []
    acc_dem = 0
    for dem_step in dhw_demand:
        acc_dem += dem_step
        dhw_demand_sumline.append(acc_dem)
    storage_load_sumline = []
    acc_load = 0
    for stor_step in storage_load:
        acc_load += stor_step
        storage_load_sumline.append(acc_load)
    storage_load_sumline = [Q + Q_full for Q in storage_load_sumline]

    # add Difference to the Last 0 Entry of the Time Series -> Sum of demands is the same then
    fill_last_zero_index_with_demand_diff = False
    if fill_last_zero_index_with_demand_diff:
        last_zero_index = None
        for idx, item in enumerate(reversed(storage_load), start=0):
            if item == 0:
                last_zero_index = idx
        storage_load[last_zero_index] += diff

    # plot Summenlinien
    if plot_demand:
        plt.plot([dem_step/(3600*1000) for dem_step in dhw_demand_sumline])
        plt.plot([stor_step/(3600*1000) for stor_step in storage_load_sumline])
        plt.ylabel('storage load and dhw demand in kWh')
        plt.title('Bedarfs ({} Peaks) und Versorgungskennliene ({} Peaks)'.format(dhw_peaks, stor_peaks))
        plt.show()

    # Input Unit of DHW demand should be equal to Output Unit of DHW demand
    storage_load = [stor_step/s_step for stor_step in storage_load]

    return storage_load


def generate_dhw_profile_pycity(plot_demand=False):
    """
    from https://github.com/RWTH-EBC/pyCity
    Problem: a lot of parameters, every time a different dhw profile
    :return:
    """
    #  Generate environment with timer, weather, and prices objects
    timer = time.Timer(time_discretization=3600,  # in seconds
                       timesteps_total=int(365 * 24)
                       )

    weather = weath.Weather(timer=timer)
    prices = price.Prices()
    environment = env.Environment(timer=timer, weather=weather, prices=prices)

    #  Generate occupancy object with stochastic user profile
    occupancy = occ.Occupancy(environment=environment, number_occupants=5)

    dhw_obj = dhw.DomesticHotWater(
        environment=environment,
        t_flow=60,  # DHW output temperature in degree Celsius
        method=2,  # Stochastic dhw profile, Method 1 not working
        supply_temperature=25,  # DHW inlet flow temperature in degree C.
        occupancy=occupancy.occupancy)  # Occupancy profile (600 sec resolution)
    dhw_demand = dhw_obj.loadcurve  # ndarray with 8760 timesteps in Watt

    if plot_demand:
        plt.plot(dhw_demand)
        plt.ylabel('dhw pycity, sum={:.2f}'.format(sum(dhw_demand) / 1000))
        plt.show()

    return dhw_demand


def import_demands_from_github(compute_cold=False, plot_demand=False):
    """
    Imports the demand series from the IBPSA Project. The time intervall is 10mins (600secs),
    therefore, the timeseries is 365*24*6 = 52560 elements long. The Unit is Watts [W].
    :param compute_cold:    switch the heat demand from summer to winter to get a sort-of-cold-series
    :param plot_demand:     plot the demand series for easy inspection
    :return: return the heat (&cold) demand as a list object in [W]
    """
    github_ibpsa_file = 'https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/' \
                        'Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv'
    demand_data = pd.read_csv(github_ibpsa_file, sep=';', index_col=0)  # first row (timesteps) is dataframe index
    demand_data.columns = demand_data.columns.str.replace(' / W', '')  # rename demand

    heat_demand_df = demand_data[["SimpleDistrict_1"]]  # take demand from one house
    heat_demand = heat_demand_df["SimpleDistrict_1"].values  # make numpy nd array
    heat_demand = [round(x, 1) for x in heat_demand]  # this demand is rounded to 1 digit for better readability

    yearly_heat_demand = sum(heat_demand)*600/(1000*3600)  # in kWh
    average_heat_flow = (sum(heat_demand)/len(heat_demand))/1000
    print("Yearly heating demand from IBPSA is {:.2f} kWh,"
          "with an average of {:.2f} kW".format(yearly_heat_demand, average_heat_flow))

    if plot_demand:
        plt.plot([dem/1000 for dem in heat_demand])
        plt.ylabel('heat demand in kW, sum={:.2f} kWh, av={:.2f} kW'.format(yearly_heat_demand, average_heat_flow))
        plt.show()

    return_series = heat_demand

    if compute_cold:
        half = int((len(heat_demand_df) - 1) / 2)  # half the length of the demand timeseries

        cold_demand_df = heat_demand_df[half:].append(heat_demand_df[:half])  # shift demand by half a year
        cold_demand_df.reset_index(inplace=True, drop=True)
        cold_demand = cold_demand_df["SimpleDistrict_1"].values  # mane numpy nd array
        cold_demand = [round(x, 1) for x in cold_demand]  # this demand is rounded to 1 digit for better readability

        yearly_cold_demand = sum(cold_demand) * 600 / (1000 * 3600)  # in kWh
        average_cold_flow = (sum(cold_demand) / len(cold_demand)) / 1000
        print("Yearly cooling demand from IBPSA is {:.2f} kWh,"
              "with an average of {:.2f} kW".format(yearly_cold_demand, average_cold_flow))

        if plot_demand:
            plt.plot(cold_demand)
            plt.ylabel('cold demand in kW, sum={:.2f} kWh, av={:.2f} kW'.format(yearly_cold_demand, average_cold_flow))
            plt.show()

        return_series = heat_demand, cold_demand

    return return_series


def import_ground_temp_table(dir_sciebo, output_interval=600, plot_series=False, start_in_summer=False):
    """
    Imports the ground Temperature file from the DWD. Data can be found at
    https://cdc.dwd.de/rest/metadata/station/html/812300083047
    hourly timesteps
    :param dir_sciebo: sciebo folder where the data is stored
    :param output_interval: Only gets upscaled, as all DWD profiles are hourly profiles(?)
    :param plot_series: decide if you want to plot the temperature series
    :return: return the series as a list object
    """

    ground_temp_file = "/demand_profiles/Soil_Temperatures/Berlin_Tempelhof_ID433/csv/data/data_TE100_MN002.csv"
    path_temp_file = dir_sciebo + ground_temp_file

    ground_temps_csv_df = pd.read_csv(path_temp_file, sep=',', index_col="Zeitstempel")

    ground_temps_df = ground_temps_csv_df[["Wert"]]
    ground_temps_np = ground_temps_df["Wert"].values  # mane numpy nd array
    ground_temps_lst = [round(x, 2) for x in ground_temps_np]  # rounded to 2 digits for better readability
    mean_temp = sum(ground_temps_lst) / len(ground_temps_lst)

    if plot_series:
        plt.plot(ground_temps_lst)
        plt.ylabel('Mean Yearly Temperature = {:.2f}'.format(mean_temp))
        plt.show()

    ground_temps_lst = [273.15 + temp for temp in ground_temps_np]  # convert to Kelvin

    # upscale series to a certain length
    steps = 3600 / output_interval
    ground_temps_lst = [[temp] * int(steps) for temp in ground_temps_lst]  # list of lists
    ground_temps_lst = [temp for temp_step in ground_temps_lst for temp in temp_step]   # flatten list

    if start_in_summer:
        half = int((len(ground_temps_lst) - 1) / 2)  # half the length of the demand timeseries
        ground_temps_lst = ground_temps_lst[half:] + ground_temps_lst[:half]  # shift demand by half a year

    return ground_temps_lst


def import_from_dhwcalc(dir_sciebo, s_step=600, delta_t_dhw=35, plot_demand=False, start_in_summer=False):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour).
    To get Energyflows, we have to multiply by rho, cp and dt. -> Q = Vdot * rho * cp * dt

    :return: dhw_demand:    time series. each timestep contains the Energyflow in Watt -> W
    """
    if s_step == 3600:
        dhw_file_in_sciebo = "/demand_profiles/DHWcalc/DHW_default_8760_200l_stepFunctionforMonths/DHW0001_DHW.txt"
    elif s_step == 600:
        dhw_file_in_sciebo = "/demand_profiles/DHWcalc/DHW_10min_200l_stepFunctionforMonths/DHW10mins_DHW.txt"
    else:
        raise Exception("Unkown Time Step for DHWcalc")
    dhw_profile = dir_sciebo + dhw_file_in_sciebo

    dhw_demand_LperH_perStep = [int(word.strip('\n')) for word in open(dhw_profile).readlines()]     # L/h each step
    dhw_demand_LperSec_perStep = [x/3600 for x in dhw_demand_LperH_perStep]

    rho = 1  # 1L = 1kg for Water
    cp = 4180  # J/kgK
    dt = delta_t_dhw  # K
    dhw_demand = [LperSec_per_step * rho * cp * dt for LperSec_per_step in dhw_demand_LperSec_perStep]  # in W

    yearly_dhw_energy_demand = sum(dhw_demand) * s_step / (3600 * 1000)     # in kWh
    average_dhw_heat_flow = (sum(dhw_demand) / len(dhw_demand)) / 1000          # in kW
    print("Yearly DHW energy demand from DHWcalc is {:.2f} kWh"
          " with an average of {:.2f} kW".format(yearly_dhw_energy_demand, average_dhw_heat_flow))

    if plot_demand:
        plt.plot(dhw_demand)
        plt.ylabel('dhw DHWcalc kWh, sum={:.2f}'.format(yearly_dhw_energy_demand))
        plt.show()

    if start_in_summer:
        half = int((len(dhw_demand) - 1) / 2)  # half the length of the demand timeseries
        dhw_demand = dhw_demand[half:] + dhw_demand[:half]  # shift demand by half a year

    return dhw_demand   # in W


def import_demands_from_demgen(dir_sciebo, house_type='Standard', output_interval=600, plot_demand=False,
                               start_in_summer=False):
    """
    files from EON.EBC DemGen. 8760 time steps in [W]
    Calculate your own demands at http://demgen.testinstanz.os-cloud.eonerc.rwth-aachen.de/
    Values for DemGen can be found in "SIA2024 - Raumnutzungsdaten Energie Gebaeudetechnik - 2015 - Page 33"

    :param dir_sciebo:      String: sciebo folder
    :param house_type:      String: Definitions of SIA2024. Standard = Standardwert, Old = Bestand, New = Zielwert
    :param output_interval: int:    Output interval. Only gets upscaled, as all  DemGen profiles are hourly profiles.
    :param plot_demand:     Bool:   decide to plot the demand profiles
    :return heat_demand:    List:   heating demand time series
            cold_demand:    List:   cooling demand time series
    """

    # Todo: API for DemGen would be very nice
    if house_type == 'Standard':
        heat_profile_file = '/demand_profiles/DemGen/Heat_demand_Berlin_200qm_SingleFamilyHouse_SIA_standard_Values.txt'
        cold_profile_file = '/demand_profiles/DemGen/Cool_demand_Berlin_200qm_SingleFamilyHouse_SIA_standard_Values.txt'
    elif house_type == 'Old':
        heat_profile_file = '/demand_profiles/DemGen/Heat_demand_Berlin_200qm_SingleFamilyHouse_SIA_Bestand_Values.txt'
        cold_profile_file = '/demand_profiles/DemGen/Cool_demand_Berlin_200qm_SingleFamilyHouse_SIA_Bestand_Values.txt'
    elif house_type == 'New':
        raise Exception("Not Implemented yet")
    else:
        raise Exception("Unknown House Type for DemGen")

    # Absolute Path
    heat_demand_file = dir_sciebo + heat_profile_file
    cold_demand_file = dir_sciebo + cold_profile_file

    # import txt file to numpy array
    heat_demand_np = np.loadtxt(heat_demand_file)
    cold_demand_np = np.loadtxt(cold_demand_file)

    # demand is rounded to 1 digit for better readability and converted to a list object
    heat_demand = [round(x, 1) for x in heat_demand_np]     # in W
    cold_demand = [round(x, 1) for x in cold_demand_np]     # in W

    # print total energy and average energy flow
    yearly_heat_demand = sum(heat_demand) / 1000    # in kWh
    average_heat_demand = (sum(heat_demand) / len(heat_demand)) / 1000  # in kW
    print("Yearly heating demand for a {} house from DemGen is {:.2f} kWh "
          "with an average of {:.2f} kW".format(house_type, yearly_heat_demand, average_heat_demand))
    yearly_cold_demand = sum(cold_demand) / 1000    # in kWh
    average_cold_demand = (sum(cold_demand) / len(cold_demand)) / 1000  # in kW
    print("Yearly cooling demand for a {} house from DemGen is {:.2f} kWh "
          "with an average of {:.2f} kW".format(house_type, yearly_cold_demand, average_cold_demand))

    if plot_demand:
        plt.plot(heat_demand)
        plt.ylabel('heat demand, sum={:.2f} kWh'.format(yearly_cold_demand))
        plt.show()
        plt.plot(cold_demand)
        plt.ylabel('heat demand, sum={:.2f} kWh'.format(yearly_cold_demand))
        plt.show()

    # stretch out the list to match a given output interval. F.e: [2,4] -> steps = 3 ->[2,2,2,4,4,4]
    steps = 3600/output_interval
    heat_demand = [[dem] * int(steps) for dem in heat_demand]   # list of lists
    heat_demand = [dem for dem_step in heat_demand for dem in dem_step]  # flatten list
    cold_demand = [[dem] * int(steps) for dem in cold_demand]  # list of lists
    cold_demand = [dem for dem_step in cold_demand for dem in dem_step]  # flatten lsit

    if start_in_summer:
        half = int((len(heat_demand) - 1) / 2)  # half the length of the demand timeseries
        heat_demand = heat_demand[half:] + heat_demand[:half]  # shift demand by half a year
        cold_demand = cold_demand[half:] + cold_demand[:half]  # shift demand by half a year

    return heat_demand, cold_demand     # in W


def create_study_pre_csv(dir_sciebo):
    dir_sciebo = Path(dir_sciebo)
    dir_models = Path(dir_sciebo/'models')
    study_pre_csv = Path(dir_models / 'study_pre.csv')
    csv_files = find("*overview.csv", dir_models)
    if len(csv_files) == 0:
        raise Exception("No overview.csv files were found in the models folder")
    else:
        # if theres yet no study.csv, the first overview.csv from the models folder is copied and renamed to study.csv
        if not study_pre_csv.is_file():
            shutil.copy(csv_files[0], dir_models)
            first_csv_name = Path(csv_files[0]).name
            first_study_csv = Path(dir_models / first_csv_name)
            first_study_csv.rename(study_pre_csv)
            del csv_files[0]  # for not appending it again
        else:
            raise Exception("A study_pre.csv file was already found in the models folder."
                            "Please clean your models folder before starting a new simulation study.")

        study_pre_df = pd.read_csv(study_pre_csv, index_col=0)

        for csv_file in csv_files:
            overview_df = pd.read_csv(csv_file, index_col=0)
            study_pre_df = pd.concat([study_pre_df, overview_df], axis='rows')
            study_pre_df.drop_duplicates()
            study_pre_df.sort_index(axis='index', ascending=False, inplace=True)

        print(study_pre_df.head())
        study_pre_df.to_csv(study_pre_csv)


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
        print("No File found that contains '{pattern}' "
              "in result directory {path}".format(pattern=pattern, path=path))

    return results


def parameter_study(params_dict, dir_sciebo):
    """
    Function that takes the params_dict and creates all possible combinations of dictionaries, depending on how many
    parameters are given. Each alternative dictionary is then passed to the 'generate_model' function.
    The number of generated dictionaries, and thus the number of simulations is returned.
    In order to no confuse time-series with parameters (for example to pass a demand profile),
    only value lists with less than 1000 entries are considered parameters.

    :param params_dict: Dictionary: simulation parameters and their initial values
    :param dir_sciebo:  String:     path to the sciebo folder
    :return: len(param_grid):       number of created simulations
    """
    # lists that have only have 1 entry aren't considered parameters, but constants.
    # lists with more than 1000 entries are considered time series
    params_dict_lsts = {k: v for k, v in params_dict.items()
                        if type(v) in [list, np.ndarray] and len(v) < 1000 and len(v) != 1}

    params_dict_series = {k: v for k, v in params_dict.items()
                          if type(v) in [list, np.ndarray] and len(v) > 1000}

    params_dict_cnsts = {}
    for k, v in params_dict.items():
        if type(v) in [str, int, float, np.float64]:
            params_dict_cnsts[k] = v
        if type(v) == list and len(v) == 1:
            params_dict_cnsts[k] = v[0]  # lists with one entry arent passed on as list objects

    # A SciKit ParameterGrid creates a generator over which you can iterate in a normal for loop.
    # In each iteration, you will have a dictionary containing the current parameter combination.
    param_grid = ParameterGrid(params_dict_lsts)

    print("There are {} values given: {} constants, {} time-series"
          " and {} parameters with a total of {} associated values. \n"
          "This results in {} combinations of the given parameters"
          .format(len(params_dict), len(params_dict_cnsts), len(params_dict_series), len(params_dict_lsts),
                  sum([len(x) for x in params_dict_lsts.values()]), len(param_grid)))

    if len(param_grid) >= 30:
        while True:  # while loop runs always, till break point
            yes_or_no = input("The parameter grid bigger than 30, do you want to continue? [y/n]")
            if yes_or_no == 'y':
                break
            elif yes_or_no == 'n':
                sys.exit("Parameter study stopped by user")
            else:
                print("you have made an invalid choice, type 'y' or 'n'.")

    for combi_dict in param_grid:
        final_dict = {**params_dict_cnsts, **params_dict_series, **combi_dict}
        # print(final_dict.keys)
        generate_model(params_dict=final_dict, dir_sciebo=dir_sciebo)

    create_study_pre_csv(dir_sciebo)

    return len(param_grid)


def generate_model(params_dict, dir_sciebo, s_step=600, save_params_to_csv=True):
    """
    Defines a building node dictionary and adds it to the graph. Adds network nodes to the graph and creates
    a district heating network graph. Creates a Modelica Model of the Graph and saves it to dir_sciebo.
    :param params_dict: dictionary:         simulation parameters and their initial values
    :param dir_sciebo:  string:             path of the sciebo folder
    :param s_step:      Integer:            Timestep Length in Seconds
    :param save_params_to_csv: boolean:    defines if parameter dict is saved to csv for later analysis
    :return:
    """
    # -------------------------------------- Network Layout -----------------------------------------------
    # Node Data has, X, Y coordinates and the peak load (supply, demand and pipe nodes!)
    node_data = pd.read_csv('Node data.csv', sep=',')
    node_data = node_data.set_index('Node')

    simple_district = ug.UESGraph()  # creates empty uesGraph Object
    # add the Supply Node (position could als be taken from the node_data csv)
    simple_district.add_building(name="Destest_Supply", position=Point(44.0, -12.0), is_supply_heating=True)  #

    # Add building and network nodes from node_data to the uesGraph object
    for node_name, values in node_data.iterrows():
        if node_name.startswith("Simple"):  # all Simple_districts are added as buildings
            simple_district.add_building(
                name=node_name,
                position=Point(values['X-Position [m]'], values['Y-Position [m]']),
                is_supply_heating=False)
        else:
            simple_district.add_network_node(
                'heating',
                name=node_name,
                position=Point(values['X-Position [m]'], values['Y-Position [m]']),
                is_supply_heating=False)

    # Help dictionary for drawing the connections / edges 16 buildings
    # total of 24 connections/pipes (each network nodes is connected to 3 other nodes)
    # node 'i' is not in here, but node 'Destest_supply'
    connection_dict_heating_nodes = {
        "a": ["b", "SimpleDistrict_2", "SimpleDistrict_3"],
        "b": ["c", "SimpleDistrict_5", "SimpleDistrict_6"],
        "c": ["d", "SimpleDistrict_10", "SimpleDistrict_11"],
        "g": ["h", "SimpleDistrict_9", "SimpleDistrict_12"],
        "d": ["Destest_Supply", "SimpleDistrict_16", "SimpleDistrict_15"],
        "e": ["f", "SimpleDistrict_1", "SimpleDistrict_4"],
        "f": ["g", "SimpleDistrict_8", "SimpleDistrict_7"],
        "h": ["Destest_Supply", "SimpleDistrict_14", "SimpleDistrict_13"],
    }

    # Adding the connections to the uesgraph object as edges
    for key, values in connection_dict_heating_nodes.items():
        for value in values:
            simple_district.add_edge(simple_district.nodes_by_name[key], simple_district.nodes_by_name[value])

    # adding info to the graph, part of this could also be done with the prepare_graph function
    # afterwards, the xxx__ pre-strings are removed from the dictionary keywords.
    for params_dict_key in params_dict.keys():

        # ----------------------------------------- General/Graph Data ------------------------------------------------
        if params_dict_key.startswith("graph__"):
            simple_district.graph[params_dict_key.replace('graph__', '')] = params_dict[params_dict_key]

        # ------------------------------------------- House/Demand Data -----------------------------------------------
        elif params_dict_key.startswith("demand__"):
            for bldg in simple_district.nodelist_building:
                if not simple_district.nodes[bldg]['is_supply_heating']:
                    simple_district.nodes[bldg][params_dict_key.replace('demand__', '')] = params_dict[params_dict_key]
        # ----------------------------------- Supply/Balancing Unit Data ----------------------------------------------
        elif params_dict_key.startswith("supply__"):
            for bldg in simple_district.nodelist_building:
                if simple_district.nodes[bldg]['is_supply_heating']:
                    simple_district.nodes[bldg][params_dict_key.replace('supply__', '')] = params_dict[params_dict_key]

    # ----------------------------------------- Pipe/Edge Data -------------------------------------------------------
    # pipe data has the beginning and ending node, Length, Diameter, thickness, peak load, pressure loss, U-value
    pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')  # rename node 'i' to 'Destest_Supply'

    # Add Diameter[m], Length[m], Insulation Thickness[m] and U-Value [W/mK] to edges/pipes
    for index, row in pipe_data.iterrows():
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['diameter'] \
    #         = row['Inner Diameter [m]'] * params_dict["pipe_scaling_factor"]
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['dh'] \
    #         = row['Inner Diameter [m]'] * params_dict["pipe_scaling_factor"]
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['length'] = row['Length [m]']
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['dIns'] = row['Insulation Thickness [m]']
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['kIns'] = row['U-value [W/mK]']

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = str(edge[0]) + 'to' + str(edge[1])

        for params_dict_key in params_dict.keys():
            if params_dict_key.startswith("edge__"):
                simple_district.edges[edge[0], edge[1]][
                    params_dict_key.replace('edge__', '')] = params_dict[params_dict_key]

    # size pipe diameter
    simple_district = utils.size_hydronic_network(
        graph=simple_district,
        network_type="heating",
        delta_t_heating=params_dict['demand__dT_Network'],
        dp_set=params_dict['size_dp_set'],   # tabellen von Ziyuan betrachten
        loop=False)

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['dh'] = simple_district.edges[edge[0], edge[1]]['diameter']


    # write m_flow_nominal to the graphs edges with uesgraph function
    sysmod_utils.estimate_m_flow_nominal(graph=simple_district, dT_design=10,
                                         network_type='heating', input_heat_str=params_dict['input_heat_str'])

    # get the maximum m_flow_nominal from the edges (set by the estimate_m_flow_nominal function)
    m_flow_nominal_max = 0
    for edge in simple_district.edges():
        m_flow_nominal_edge = simple_district.edges[edge[0], edge[1]]["m_flow_nominal"]

        if m_flow_nominal_edge > m_flow_nominal_max:
            m_flow_nominal_max = m_flow_nominal_edge

    # write m_flow_nominal to the supply
    # Todo: no support for multiple supplies yet
    pipes_from_supply = 2
    safety_factor = 1
    m_flow_nominal_supply = m_flow_nominal_max * pipes_from_supply * safety_factor
    for bldg in simple_district.nodelist_building:
        if simple_district.nodes[bldg]['is_supply_heating']:
            simple_district.nodes[bldg]["m_flow_nominal"] = m_flow_nominal_supply

    # --------------- Visualization, Save  ----------------
    vis = ug.Visuals(simple_district)  # Plotting / Visualization with pipe diameters scaling
    vis.show_network(
        save_as="uesgraph_destest_16_selfsized_jonas.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100)

    dir_models = os.path.join(dir_sciebo, 'models')
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)

    save_name = "Destest_Jonas_{}".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))  # for unique naming
    dir_model = os.path.join(dir_models, save_name)

    # ------------modified create_model function, default is part of sysmod_utils -------------
    assert not save_name[0].isdigit(), "Model name cannot start with a digit"

    new_model = sysmh.SystemModelHeating(network_type=simple_district.graph["network_type"])
    new_model.stop_time = 365 * 24 * 3600
    new_model.timestep = s_step
    new_model.import_from_uesgraph(simple_district)
    new_model.set_connection(remove_network_nodes=True)

    new_model.add_return_network = True
    new_model.medium = params_dict['model_medium']
    new_model.medium_building = params_dict['model_medium']
    new_model.T_nominal = params_dict['graph__t_nominal']
    new_model.p_nominal = params_dict['graph__p_nominal']

    new_model.set_control_pressure(
        name_supply=params_dict['pressure_control_supply'],
        dp=params_dict['pressure_control_dp'],
        name_building=params_dict['pressure_control_building'],
        p_max=params_dict['pressure_control_p_max'],
        k=params_dict['pressure_control_k'],
        ti=params_dict['pressure_control_ti'],
    )

    new_model.ground_model = "t_ground_table"
    new_model.graph["T_ground"] = params_dict['graph__t_ground']
    new_model.tolerance = 1e-5

    for node in new_model.nodelist_building:
        is_supply = "is_supply_{}".format(new_model.network_type)
        if new_model.nodes[node][is_supply]:
            new_model.nodes[node]["comp_model"] = params_dict['model_supply']
        else:
            new_model.nodes[node]["comp_model"] = params_dict['model_demand']

    for node in new_model.nodelist_pipe:
        new_model.nodes[node]["comp_model"] = params_dict['model_pipe']

    name = save_name[0].upper() + save_name[1:]
    new_model.model_name = name
    new_model.write_modelica_package(save_at=dir_models)
    # ---------------------------- end create_model function ---------------------------------

    if save_params_to_csv:
        path_to_csv = dir_model + "/" + save_name + "_overview.csv"
        # if time series are inside the params_dict, only the first entry is passed to the overview.csv
        params_lst_series = [k for k, v in params_dict.items() if type(v) in [list, np.ndarray] and len(v) > 1000]
        for series_i in params_lst_series:
            new_entry = 'time-series, starts with: ' + str(params_dict[series_i][0])
            params_dict[series_i] = new_entry
        overview_df = pd.DataFrame.from_records(params_dict, index=[save_name])
        overview_df.to_csv(path_to_csv)


if __name__ == '__main__':
    main()

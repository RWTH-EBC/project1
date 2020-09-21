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
import itertools
from sklearn.model_selection import ParameterGrid
import sys
import matplotlib.pyplot as plt

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
    ground_temps = import_ground_temp_table(dir_sciebo, plot_series=False)

    # dhw_demand_pycity = generate_dhw_profile_pycity()
    dhw_demand_dhwcalc = import_from_dhwcalc(dir_sciebo, plot_demand=False)
    # heat_demand_ibpsa = import_demands_from_github()
    heat_demand_demgen, cold_demand_demgen = import_demands_from_demgen(dir_sciebo, plot_demand=False)

    heat_demand = heat_demand_demgen
    cold_demand = cold_demand_demgen
    dhw_demand = dhw_demand_dhwcalc

    max_heat_demand = max(heat_demand)
    max_cold_demand = max(cold_demand)

    # ------------------------ Parameter Dictionaries to pass into parameter study function -------------------------
    aixlib_dhc = "AixLib.Fluid.DistrictHeatingCooling."
    params_dict_testing = {
        # 'variable': 'value',      # eventually needed for csv generation? pandas takes first entry as index
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',  # needs ground temperature?
        # 'model_pipe': "AixLib.Fluid.FixedResistances.PlugFlowPipe",
        # 'model_pipe': aixlib_dhc + 'Pipes.StaticPipe',
        'fac': 1.0,
        'roughness': 2.5e-5,
        # ----------------------- General Node Data ---------------------
        't_return': 273.15 + 35,  # function of T_supply and dT_design? -> redundant? no(!)
        # -------------------------- Demand/House Node Data ----------------------------
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.HeatPumpCarnot',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDp',  # aus E11
        # 'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledHeatPumpFixDeltaT",  # Erdeis
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpFixDeltaT",
        'dT_design': 10,
        't_nominal': 273.15 + 25,  # equals T_Ambient in Dymola? Start value for every pipe?
        't_ground': 273.15 + 10,
        'p_nominal': 5e5,
        'm_flo_bypass': 0.0005,
        'dT_building': 10,  # inside the buildings? necessary for heatpump demand models
        'cop_nominal': 5.0,
        't_supply_building': 273.15 + 40,  # should be higher than T condensator? necessary for heatpump demand models
        't_con_nominal': 273.15 + 35,
        't_eva_nominal': 273.15 + 10,  # should be around ground temp?
        # 'dTEva_nominal': dTEva_nominal,
        # 'dTCon_nominal': dTCon_nominal,
        # ------------------------------ Supply Node Data --------------------------
        # 'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdeal',
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
        # 'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdealPump',
        't_supply': 273.15 + 45,  # -> TIn in Modelica
        'p_supply': 6e5,
        'p_return': 2e5,
        'm_flow_nominal_supply': 1.0,
        # ---------- further create_model data ---------
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        'model_ground': "t_ground_table",
    }
    params_dict_study_models = {
        # 'variable': 'value',      # eventually needed for csv generation? pandas takes first entry as index
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': [
            aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
            "AixLib.Fluid.FixedResistances.PlugFlowPipe",
            aixlib_dhc + 'Pipes.StaticPipe'
        ],
        'fac': 1.0,
        'roughness': 2.5e-5,
        # ----------------------- General Node Data ---------------------
        't_return': 273.15 + 35,  # function of T_supply and dT_design? -> redundant? no(!)
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': [
            aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpFixDeltaT",
            aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
            aixlib_dhc + 'Demands.OpenLoop.HeatPumpCarnot',
            aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDp',  # aus E11
            # aixlib_dhc + "Demands.ClosedLoop.PumpControlledHeatPumpFixDeltaT",  # Erdeis
        ],
        'dT_design': 10,
        't_nominal': 273.15 + 25,  # equals T_Ambient in Dymola? Start value for every pipe?
        't_ground': 273.15 + 10,
        'p_nominal': 5e5,
        'm_flo_bypass': 0.0005,
        'dT_building': 10,  # inside the buildings? necessary for heatpump demand models
        'cop_nominal': 5.0,
        't_supply_building': 273.15 + 40,  # should be higher than T condensator? necessary for heatpump demand models
        't_con_nominal': 273.15 + 35,
        't_eva_nominal': 273.15 + 10,  # should be around ground temp?
        # 'dTEva_nominal': dTEva_nominal,
        # 'dTCon_nominal': dTCon_nominal,
        # ------------------------------ Supply Node Data --------------------------
        'model_supply': [
            aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
            aixlib_dhc + 'Supplies.OpenLoop.SourceIdeal',
            aixlib_dhc + 'Supplies.OpenLoop.SourceIdealPump'
        ],
        't_supply': 273.15 + 45,  # -> TIn in Modelica
        'p_supply': 6e5,
        'p_return': 2e5,
        'm_flow_nominal_supply': 1.0,
        # ---------- further create_model data ---------
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        'model_ground': "t_ground_table",
    }
    params_dict_study1 = {
        # 'variable': 'value',      # eventually needed for csv generation? pandas takes first entry as index
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',  # needs ground temperature?
        # 'model_pipe': "AixLib.Fluid.FixedResistances.PlugFlowPipe",
        # 'model_pipe': aixlib_dhc + 'Pipes.StaticPipe',
        'fac': 1.0,
        'roughness': 2.5e-5,
        # ----------------------- General Node Data ---------------------
        't_return': 273.15 + 35,  # function of T_supply and dT_design? -> redundant? no(!)
        # -------------------------- Demand/House Node Data ----------------------------
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.HeatPumpCarnot',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDp',  # aus E11
        # 'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledHeatPumpFixDeltaT",  # Erdeis
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpFixDeltaT",
        'demand': heat_demand,
        'dT_design': 10,
        't_nominal': 273.15 + 25,  # equals T_Ambient in Dymola? Start value for every pipe?
        't_ground': np.linspace(273.15 - 20.0, 273.15 + 40, 7),
        # 't_ground': [273.15 - 20.0, 273.15 - 10.0, 273.15, 273.15 + 10, 273.15 + 20, 273.15 + 30, 273.15 + 40],
        'p_nominal': 5e5,
        'm_flo_bypass': 0.0005,
        'dT_building': 10,  # inside the buildings? necessary for heatpump demand models
        'cop_nominal': np.linspace(4, 6, 3),
        # 'cop_nominal': [4, 5, 6],
        't_supply_building': 273.15 + 40,  # should be higher than T condensator? necessary for heatpump demand models
        't_con_nominal': 273.15 + 35,
        't_eva_nominal': 273.15 + 10,  # should be around ground temp?
        # 'dTEva_nominal': dTEva_nominal,
        # 'dTCon_nominal': dTCon_nominal,
        # ------------------------------ Supply Node Data --------------------------
        # 'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdeal',
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
        # 'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdealPump',
        't_supply': np.linspace(273.15 + 20, 273.15 + 45, 6),  # -> TIn in Modelica
        # 't_supply': [273.15 + 20, 273.15 + 25, 273.15 + 30, 273.15 + 35, 273.15 + 40, 273.15 + 45],  # TIn in Modelica
        'p_supply': 6e5,
        'p_return': 2e5,
        'm_flow_nominal_supply': 1.0,
        # ---------- further create_model data ---------
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        'model_ground': "t_ground_table",
    }

    params_dict_5g_heating = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 10],  # equals T_Ambient in Dymola? Start value for every pipe?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': 273.15 + 10,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,  #
        'edge__diameter': [0.1],  # Destest default: 0.02-0.05
        'edge__length': 12,  # Destest default: 12-36
        'edge__dIns': 0.04,  # Destest default: 0.045
        'edge__kIns': 0.035,  # Destest default: 0.035, U-Value
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.SubstationHeating",  # 5GDHC
        'demand__heatDemand': heat_demand,
        'demand__T_supplyHeatingSet': [273.15 + 30],  # T_VL Heizung
        # 'demand__t_return': 273.15 + 10,    # should be equal to supply__t_return!
        # 'demand__dT_design': [5, 10, 15, 20],    # needed for estimate_m_flow_nominal function
        # 'demand__m_flo_bypass': 0.0005,
        # 'demand__dT_building': 10,  # inside the buildings? necessary for heatpump demand models
        # 'demand__cop_nominal': 5.0,
        # 'demand__t_supply_building': 273.15 + 20,   # should be higher than Tcon? necessary for heatpump demand models
        # 'demand__t_con_nominal': 273.15 + 15,
        # 'demand__t_eva_nominal': 273.15 + 10,  # should be around ground temp?
        # 'dTEva_nominal': dTEva_nominal,
        # 'dTCon_nominal': dTCon_nominal,
        'demand__heatDemand_max': max_heat_demand,
        'demand__deltaT_heatingSet': [10],  # Templeraturspreizung der Heizung
        'demand__deltaT_heatingGridSet': [6],  # Difference Hot and Cold Pipe
        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlant',
        # 'supply__TIn': 273.15 + 20,  # -> t_supply
        # 'supply__t_return': 273.15 + 10,    # should be equal to demand__t_return!
        # 'supply__dpIn': [6e5],    # p_supply
        # 'supply__p_return': 2e5,
        'supply__m_flow_nominal': [2],
        'supply__T_coolingSet': [273.15 + 16],  # Set Temperature cold Pipe
        'supply__T_heatingSet': [273.15 + 22],  # Set Temperature hot Pipe
    }
    params_dict_5g_heating_cooling = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 10],  # equals T_Ambient in Dymola? Start value for every pipe?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': 273.15 + 10,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,  #
        'edge__diameter': [0.1],  # Destest default: 0.02-0.05
        'edge__length': 12,  # Destest default: 12-36
        'edge__dIns': 0.004,  # Destest default: 0.045, Isolation Thickness
        'edge__kIns': 0.0035,  # Destest default: 0.035, U-Value
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.SubstationHeatingCoolingVarDeltaT",  # 5GDHC
        'demand__heatDemand': heat_demand,
        'demand__coolingDemand': cold_demand,
        'demand__T_supplyHeatingSet': [273.15 + 30],  # T_VL Heizung
        'demand__T_supplyCoolingSet': [273.15 + 12],  # T_VL Kühlung
        'demand__heatDemand_max': max_heat_demand,
        'demand__coolingDemand_max': max_cold_demand,

        'demand__deltaT_heatingSet': [5],  # Templeraturspreizung der Heizung
        'demand__deltaT_coolingSet': [5],  # Templeraturspreizung der Kühlung

        'demand__deltaT_heatingGridSet': [4],  # Difference Hot and Cold Pipe
        'demand__deltaT_coolingGridSet': [4],  # Difference Cold and Hot Pipe ?????
        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlant',
        # 'supply__TIn': 273.15 + 20,  # -> t_supply
        # 'supply__t_return': 273.15 + 10,    # should be equal to demand__t_return!
        # 'supply__dpIn': [6e5],    # p_supply
        # 'supply__p_return': 2e5,
        'supply__m_flow_nominal': [2],
        'supply__T_coolingSet': [273.15 + 16],  # Set Temperature cold Pipe
        'supply__T_heatingSet': [273.15 + 22],  # Set Temperature hot Pipe
    }
    params_dict_5g_heating_cooling_xiyuan_single = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 10],  # equals T_Ambient in Dymola? Start value for every pipe?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': ground_temps,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledwithHP_v4_ze_jonas",  # 5GDHC
        'demand__heat_input': heat_demand,
        'input_heat_str': 'heat_input',
        'demand__cold_input': cold_demand,
        'demand__dhw_input': dhw_demand,
        'demand__T_dhw_supply': [273.15 + 65],  # T_VL DHW
        'demand__dT_Network': [10],
        'demand__heatDemand_max': max_heat_demand,
        # 'demand__coolingDemand_max': max_cold_demand,

        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
        'supply__TIn': [273.15 + 20],  # -> t_supply
        # 'supply__t_return': 273.15 + 10,    # should be equal to demand__t_return!
        'supply__dpIn': [5e5],  # p_supply
        # 'supply__p_return': 2e5,
        'supply__m_flow_nominal': [2],
    }
    params_dict_5g_heating_cooling_xiyuan_study = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 5, 273.15 + 10, 273.15 + 15, 273.15 + 20],  # equals T_Ambient in Dymola? Start value for every pipe?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': ground_temps,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledwithHP_v4_ze_jonas",  # 5GDHC
        'demand__heat_input': heat_demand,
        'input_heat_str': 'heat_input',
        'demand__cold_input': cold_demand,
        'demand__dhw_input': dhw_demand,
        'demand__T_dhw_supply': [273.15 + 65],  # T_VL DHW
        'demand__dT_Network': [10],
        'demand__heatDemand_max': max_heat_demand,
        # 'demand__coolingDemand_max': max_cold_demand,

        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlantPump',
        'supply__TIn': [273.15 + 15],  # -> t_supply
        # 'supply__t_return': 273.15 + 10,    # should be equal to demand__t_return!
        'supply__dpIn': [5e5, 2e5],  # p_supply
        # 'supply__p_return': 2e5,
        'supply__m_flow_nominal': [1, 2],
    }

    params_dict_5g_heating_micha = {
        # ----------------------------------------- General/Graph Data -------------------------------------------------
        'graph__network_type': 'heating',
        'graph__t_nominal': [273.15 + 10],  # equals T_Ambient in Dymola? Start value for every pipe?
        'graph__p_nominal': [1e5],
        'model_ground': "t_ground_table",
        'graph__t_ground': 273.15 + 10,
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        # ----------------- Pipe/Edge Data ----------------
        'model_pipe': aixlib_dhc + 'Pipes.PlugFlowPipeEmbedded',
        'edge__fac': 1.0,
        'edge__roughness': 2.5e-5,  #
        'edge__diameter': [0.1],  # Destest default: 0.02-0.05
        'edge__length': 12,  # Destest default: 12-36
        'edge__dIns': 0.004,  # Destest default: 0.045
        'edge__kIns': 0.0035,  # Destest default: 0.035, U-Value
        # -------------------------- Demand/House Node Data ----------------------------
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledHeatPumpFixDeltaT",  # 5GDHC
        'demand__Q_flow_input': heat_demand,
        'demand__TSupplyBuilding': [273.15 + 30],  # T_VL Heizung
        'demand__Q_flow_nominal': max_heat_demand,
        'demand__TReturn': [18],  # Return Temp vom Netz??
        'demand__dTDesign': [4],  # Difference Hot and Cold Pipe
        'demand__dTBuilding': [5],  # Temperaturspreizung Heizung
        # ------------------------------ Supply Node Data --------------------------
        'model_supply': aixlib_dhc + 'Supplies.ClosedLoop.IdealPlant',
        # 'supply__TIn': 273.15 + 20,  # -> t_supply
        # 'supply__t_return': 273.15 + 10,    # should be equal to demand__t_return!
        # 'supply__dpIn': [6e5],    # p_supply
        # 'supply__p_return': 2e5,
        'supply__m_flow_nominal': [2],
        'supply__T_coolingSet': [273.15 + 18],  # Set Temperature cold Pipe
        'supply__T_heatingSet': [273.15 + 22],  # Set Temperature hot Pipe
    }

    # ---------------------------------------- create Simulations ----------------------------------------------
    parameter_study(params_dict_5g_heating_cooling_xiyuan_study, dir_sciebo)


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


def import_demands_from_github(compute_cold=False):
    # compute cold just swaps the heat time series from winter to summer
    github_ibpsa_file = 'https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/' \
                        'Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv'
    demand_data = pd.read_csv(github_ibpsa_file, sep=';', index_col=0)  # first row (timesteps) is dataframe index
    demand_data.columns = demand_data.columns.str.replace(' / W', '')  # rename demand

    heat_demand_df = demand_data[["SimpleDistrict_1"]]

    half = int((len(heat_demand_df) - 1) / 2)  # half the length of the demand timeseries

    cold_demand_df = heat_demand_df[half:].append(heat_demand_df[:half])  # shift demand by half a year
    cold_demand_df.reset_index(inplace=True, drop=True)

    heat_demand = heat_demand_df["SimpleDistrict_1"].values  # mane numpy nd array
    cold_demand = cold_demand_df["SimpleDistrict_1"].values  # mane numpy nd array

    heat_demand = [round(x, 1) for x in heat_demand]  # this demand is rounded to 1 digit for better readability
    cold_demand = [round(x, 1) for x in cold_demand]  # this demand is rounded to 1 digit for better readability

    if not compute_cold:
        return heat_demand
    else:
        return heat_demand, cold_demand


def import_ground_temp_table(dir_sciebo, plot_series=False):
    """
    Imports the ground Temperature file from the DWD. Data can be found at
    https://cdc.dwd.de/rest/metadata/station/html/812300083047
    :param dir_sciebo: sciebo folder where the data is stored
    :param plot_series: decide if you want to plot the temperature series
    :return: return the series as a list object
    """

    ground_temp_file = "/demand_profiles/Soil_Temperatures/Berlin_Tempelhof_ID433/csv/data/data_TE100_MN002.csv"
    path_temp_file = dir_sciebo + ground_temp_file

    ground_temps_csv_df = pd.read_csv(path_temp_file, sep=',', index_col="Zeitstempel")

    ground_temps_df = ground_temps_csv_df[["Wert"]]
    ground_temps_np = ground_temps_df["Wert"].values  # mane numpy nd array
    ground_temps_lst = [round(x, 1) for x in ground_temps_np]  # this demand is rounded to 1 digit for better readability
    mean_temp = sum(ground_temps_lst) / len(ground_temps_lst)

    if plot_series:
        plt.plot(ground_temps_lst)
        plt.ylabel('Mean Yearly Temperature = {:.2f}'.format(mean_temp))
        plt.show()

    ground_temps_lst = [273.15 + temp for temp in ground_temps_np]  # convert to Kelvin

    return ground_temps_lst


def import_from_dhwcalc(dir_sciebo, delta_t_dhw=35, plot_demand=False):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour) for 8760 hourly steps.
    To get Energyflows, we have to multiply by rho, cp and dt. -> Q = Vdot * rho * cp * dt
    :return: dhw_demand:    time series. each hourly timestep contains the Energydemand in Watthours -> Wh/1h
    """

    dhw_file_in_sciebo = "/demand_profiles/DHWcalc/DHW_default_8760_200l_stepFunctionforMonths/DHW0001_DHW.txt"
    dhw_profile = dir_sciebo + dhw_file_in_sciebo

    dhw_demand_LperH = [int(word.strip('\n')) for word in open(dhw_profile).readlines()]
    dhw_demand_LperH = [round(x, 1) for x in dhw_demand_LperH]

    joule_in_Wh = 1/3600   # 1J = 1Ws = 1Wh/3600
    rho = 1  # 1L =0.001m^3 = 1kg for Water
    cp = 4180  # J/kgK
    dt = delta_t_dhw  # K

    dhw_demand = [x * rho * cp * dt * joule_in_Wh for x in dhw_demand_LperH]    # in Wh
    yearly_dhw_energy_demand = sum(dhw_demand)/1000     # in kWh
    print("Yearly DHW energy demand from DHWcalc is {:.2f} kWh".format(yearly_dhw_energy_demand))

    if plot_demand:
        plt.plot(dhw_demand)
        plt.ylabel('dhw DHWcalc kWh, sum={:.2f}'.format(yearly_dhw_energy_demand))
        plt.show()

    return dhw_demand   # in Wh


def import_demands_from_demgen(dir_sciebo, plot_demand=False):
    # files from EON.EBC DemGen. 8760 time steps in [Wh]
    # Calculate your own demands at http://demgen.testinstanz.os-cloud.eonerc.rwth-aachen.de/

    heat_profile_file = '/demand_profiles/DemGen/Heat_demand_Berlin_200qm_SingleFamilyHouse_SIA_standard_Values.txt'
    cold_profile_file = '/demand_profiles/DemGen/Cool_demand_Berlin_200qm_SingleFamilyHouse_SIA_standard_Values.txt'
    heat_demand_file = dir_sciebo + heat_profile_file
    cold_demand_file = dir_sciebo + cold_profile_file

    # import txt file to numpy array
    heat_demand_np = np.loadtxt(heat_demand_file)
    cold_demand_np = np.loadtxt(cold_demand_file)

    # demand is rounded to 1 digit for better readability and converted to a list object
    heat_demand = [round(x, 1) for x in heat_demand_np]     # in Wh
    cold_demand = [round(x, 1) for x in cold_demand_np]     # in Wh

    yearly_heat_demand = sum(heat_demand) / 1000    # in kWh
    print("Yearly heat energy demand from DemGen is {:.2f} kWh".format(yearly_heat_demand))
    yearly_cold_demand = sum(cold_demand) / 1000    # in kWh
    print("Yearly cold energy demand from DemGen is {:.2f} kWh".format(yearly_cold_demand))

    if plot_demand:
        plt.plot(heat_demand)
        plt.ylabel('heat demand, sum={:.2f}'.format(yearly_cold_demand))
        plt.show()
        plt.plot(cold_demand)
        plt.ylabel('heat demand, sum={:.2f}'.format(yearly_cold_demand))
        plt.show()

    return heat_demand, cold_demand


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

    return len(param_grid)


def generate_model(params_dict, dir_sciebo, save_params_to_csv=True):
    """
    Defines a building node dictionary and adds it to the graph. Adds network nodes to the graph and creates
    a district heating network graph. Creates a Modelica Model of the Graph and saves it to dir_sciebo.
    :param params_dict: dictionary:         simulation parameters and their initial values
    :param dir_sciebo:  string:             path of the sciebo folder
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
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['diameter'] = row['Inner Diameter [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['dh'] = row['Inner Diameter [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['length'] = row['Length [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['dIns'] = row['Insulation Thickness [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['kIns'] = row['U-value [W/mK]']

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = str(edge[0]) + 'to' + str(edge[1])
        # simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1   # part prepare_graph or estimate_m_flow_nom

        for params_dict_key in params_dict.keys():
            if params_dict_key.startswith("edge__"):
                simple_district.edges[edge[0], edge[1]][
                    params_dict_key.replace('edge__', '')] = params_dict[params_dict_key]

    # write m_flow_nominal to the graphs edges with uesgraph function
    sysmod_utils.estimate_m_flow_nominal(graph=simple_district, dT_design=10,
                                         network_type='heating', input_heat_str=params_dict['input_heat_str'])

    # --------------- Visualization, Save  ----------------
    vis = ug.Visuals(simple_district)  # Plotting / Visualization with pipe diameters scaling
    vis.show_network(
        save_as="uesgraph_destest_16_selfsized_jonas.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100)

    dir_models_sciebo = os.path.join(dir_sciebo, 'models')
    if not os.path.exists(dir_models_sciebo):
        os.mkdir(dir_models_sciebo)

    save_name = "Destest_Jonas_{}".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))  # for unique naming
    dir_model = os.path.join(dir_models_sciebo, save_name)

    # creates a return network, creates a model from SystemModelHeating, sets the demand, supply, pipe etc. models
    # sysmod_utils.create_model(
    #     name=save_name,
    #     save_at=dir_models_sciebo,
    #     graph=simple_district,
    #     stop_time=365 * 24 * 3600,
    #     timestep=600,
    #     model_supply=params_dict['model_supply'],
    #     model_demand=params_dict['model_demand'],
    #     model_pipe=params_dict['model_pipe'],
    #     model_medium=params_dict['model_medium'],
    #     model_ground=params_dict['model_ground'],
    #     T_nominal=params_dict['graph__t_nominal'],
    #     p_nominal=params_dict['graph__p_nominal'],
    #     t_ground_prescribed=params_dict['graph__t_ground']
    # )

    # ------------modified create_model function, default is part of sysmod_utils -------------
    assert not save_name[0].isdigit(), "Model name cannot start with a digit"

    new_model = sysmh.SystemModelHeating(network_type=simple_district.graph["network_type"])
    new_model.stop_time = 365 * 24 * 1
    new_model.timestep = 1
    new_model.import_from_uesgraph(simple_district)
    new_model.set_connection(remove_network_nodes=True)

    new_model.add_return_network = True
    new_model.medium = params_dict['model_medium']
    new_model.medium_building = params_dict['model_medium']
    new_model.T_nominal = params_dict['graph__t_nominal']
    new_model.p_nominal = params_dict['graph__p_nominal']
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
    new_model.write_modelica_package(save_at=dir_models_sciebo)
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

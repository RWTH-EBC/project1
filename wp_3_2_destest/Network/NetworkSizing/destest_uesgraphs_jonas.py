# -*- coding: utf-8 -*-
import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import numpy as np
import os
import platform
from datetime import datetime
from uesgraphs.systemmodels import utilities as sysmod_utils
import itertools
from sklearn.model_selection import ParameterGrid
import sys


# import csv


def main():
    # paths
    if platform.system() == 'Darwin':
        dir_sciebo = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    elif platform.system() == 'Windows':
        dir_sciebo = "D:/mma-jgr/sciebo-folder/RWTH_Dokumente/MA_Masterarbeit_RWTH/Data"
    else:
        raise Exception("Unknown operating system")

    # parameters
    aixlib_dhc = "AixLib.Fluid.DistrictHeatingCooling."

    heat_demand, cold_demand = import_demands_from_github()
    max_heat_demand = max(heat_demand)
    max_cold_demand = max(cold_demand)

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
        'demand__dTBuilding': [5],   # Temperaturspreizung Heizung
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

    parameter_study(params_dict_5g_heating_cooling, dir_sciebo)


def import_demands_from_github():
    demand_data = pd.read_csv('https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/'
                              + 'Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv',
                              sep=';',
                              index_col=0)  # first row, the timesteps are taken as the dataframe index
    demand_data.columns = demand_data.columns.str.replace(' / W', '')  # rename demand

    heat_demand_df = demand_data[["SimpleDistrict_1"]]

    half = int((len(heat_demand_df) - 1) / 2)   # half the length of the demand timeseries

    cold_demand_df = heat_demand_df[half:].append(heat_demand_df[:half])     # shift demand by half a year
    cold_demand_df.reset_index(inplace=True, drop=True)

    heat_demand = heat_demand_df["SimpleDistrict_1"].values     # mane numpy nd array
    cold_demand = cold_demand_df["SimpleDistrict_1"].values     # mane numpy nd array

    heat_demand = [round(x, 1) for x in heat_demand]  # this demand is rounded to 1 digit for better readability
    cold_demand = [round(x, 1) for x in cold_demand]  # this demand is rounded to 1 digit for better readability

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
    "Defines a building node dictionary and adds it to the graph. Adds network nodes to the graph and creates
    a district heating network graph. Creates a Modelica Model of the Graph and saves it to dir_sciebo."
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
    # # pipe data has the beginning and ending node, Length, Diameter, thickness, peak load, pressure loss, U-value
    # pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    # pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')  # rename node 'i' to 'Destest_Supply'
    #
    # # Add Diameter[m], Length[m], Insulation Thickness[m] and U-Value [W/mK] to edges/pipes
    # for index, row in pipe_data.iterrows():
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['diameter'] = row['Inner Diameter [m]'] * 20
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['length'] = row['Length [m]']
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['dIns'] = row['Insulation Thickness [m]'] / 10
    #     simple_district.edges[
    #         simple_district.nodes_by_name[row['Beginning Node']],
    #         simple_district.nodes_by_name[row['Ending Node']]]['kIns'] = row['U-value [W/mK]']

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = str(edge[0]) + 'to' + str(edge[1])
        # simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1   # part prepare_graph or estimate_m_flow_nom

        for params_dict_key in params_dict.keys():
            if params_dict_key.startswith("edge__"):
                simple_district.edges[edge[0], edge[1]][
                    params_dict_key.replace('edge__', '')] = params_dict[params_dict_key]

    # write m_flow_nominal to the graphs edges with uesgraph function
    sysmod_utils.estimate_m_flow_nominal(graph=simple_district, dT_design=params_dict['demand__deltaT_heatingGridSet'],
                                         network_type='heating', input_heat_str='heatDemand')

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
    sysmod_utils.create_model(
        name=save_name,
        save_at=dir_models_sciebo,
        graph=simple_district,
        stop_time=365 * 24 * 3600,
        timestep=600,
        model_supply=params_dict['model_supply'],
        model_demand=params_dict['model_demand'],
        model_pipe=params_dict['model_pipe'],
        model_medium=params_dict['model_medium'],
        model_ground=params_dict['model_ground'],
        T_nominal=params_dict['graph__t_nominal'],
        p_nominal=params_dict['graph__p_nominal'],
        t_ground_prescribed=params_dict['graph__t_ground']
    )

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

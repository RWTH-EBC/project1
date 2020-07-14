# -*- coding: utf-8 -*-
import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import numpy as np
import os
import platform
from datetime import datetime
from uesgraphs.systemmodels import utilities as sysmod_utils
import csv


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
    params_dict = {
        # 'variable': 'value',      # eventually needed for csv generation? pandas takes first entry as index
        't_supply': 273.15 + 32,    # -> TIn in Modelica
        't_return': 273.15 + 22,    # function of T_supply and dT_design? -> redundant?
        'dT_design': 10,
        't_nominal': 273.15 + 25,   # equals T_Ambient in Dymola? Start value for every pipe?
        't_ground': 273.15 + 10,
        'p_supply': 6e5,    # dP_In -> Druckdifferenzregelung nochmal genau anschauen!
        'p_nominal': 5e5,
        'p_return': 2e5,
        'm_flo_bypass': 0.0005,
        't_con_nominal': 273.15 + 35,
        't_eva_nominal': 273.15 + 10,   # should be around ground temp?
        'dT_building': 10,  # inside the buildings? necessary for heatpump demand models
        'cop_nominal': 5.5,
        't_supply_building': 273.15 + 40,   # should be higher than T condensator? necessary for heatpump demand models
        'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdeal',
        # 'model_supply': aixlib_dhc + 'Supplies.OpenLoop.SourceIdealPump',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.HeatPumpCarnot',
        # 'model_demand': aixlib_dhc + 'Demands.OpenLoop.VarTSupplyDp',  # aus E11
        # 'model_demand': aixlib_dhc + "Demands.ClosedLoop.PumpControlledHeatPumpFixDeltaT",  # Erdeis
        'model_demand': aixlib_dhc + "Demands.ClosedLoop.ValveControlledHeatPumpFixDeltaT",
        # 'model_pipe': "AixLib.Fluid.FixedResistances.PlugFlowPipe",
        'model_pipe': aixlib_dhc + 'Pipes.StaticPipe',
        'model_medium': "AixLib.Media.Specialized.Water.ConstantProperties_pT",
        'model_ground': "t_ground_table",
    }

    parameter_study(params_dict, dir_sciebo,
                    p1='cop_nominal', p1_values=np.arange(4.0, 5.5, 1.0),
                    # p2='p_supply', p2_values=np.arange(5.0e5, 7.2e5, 0.2e5)
                    )


def parameter_study(params_dict, dir_sciebo, p1='', p1_values=np.arange(1, 1), p2='', p2_values=np.arange(1, 1)):
    """
    Function that takes the params_dict with its default values and creates a number of alternative
    dictionaries, depending on how many parameters and corresponding values are given. Each alternative dictionary is
    then passed to the 'generate_model' function. The number of generated dictionaries, and thus the number of
    simulations si returned.
    :param params_dict: dictionary: stores the simulation parameters and their initial values
    :param dir_sciebo:  string:     stores the path of the sciebo folder
    :param p1:          string:     name of parameter 1
    :param p1_values:   np array:   list of values for parameter 1
    :param p2:          string:     name of parameter 2
    :param p2_values:   np array:   list of values for parameter 2
    :return: runs:      integer:    number of simulations created
    """
    if p1 is '' and p2 is '':
        generate_model(params_dict=params_dict, dir_sciebo=dir_sciebo)
        runs = 1
    elif p1 is '':
        runs = len(p2_values)
        print("{} values for the {} parameter are given: {}".format(runs, p2, p2_values))
        for p2_value in p2_values:
            params_dict[p2] = p2_value
            generate_model(params_dict=params_dict, dir_sciebo=dir_sciebo)
    elif p2 is '':
        runs = len(p1_values)
        print("{} values for the {} parameter are given: {}".format(runs, p1, p1_values))
        for p1_value in p1_values:
            params_dict[p1] = p1_value
            generate_model(params_dict=params_dict, dir_sciebo=dir_sciebo)
    else:
        runs = len(p1_values) * len(p2_values)
        print("Two parameters are given: \n {} values for the {} parameter: {} \n"
              "{} values for the {} parameter: {} \n This results in {} combinations of the two parameters"
              .format(len(p1_values), p1, p1_values, len(p2_values), p2, p2_values, runs))
        for p1_value in p1_values:
            params_dict[p1] = p1_value
            for var2_value in p2_values:
                params_dict[p2] = var2_value
                generate_model(params_dict=params_dict, dir_sciebo=dir_sciebo)

    print("{} Simulation setups have been created and saved at {} \n"
          "Those setups are now ready to be simulated with Dymola".format(runs, dir_sciebo))
    return runs


def generate_model(params_dict, dir_sciebo, long_name=False, save_params_to_csv=True):
    """
    "Defines a building node dictionary and adds it to the graph. Adds network nodes to the graph and creates
    a district heating network graph. Creates a Modelica Model of the Graph and saves it to dir_sciebo."
    :param long_name: boolean:              defines if name of saved file is long (with info) or short (with timestamp)
    :param params_dict: dictionary:         simulation parameters and their initial values
    :param dir_sciebo:  string:             path of the sciebo folder
    :param save_params_to_csv: boolean:    defines if parameter dict is saved to csv for later analysis
    :return:
    """
    node_data = pd.read_csv('Node data.csv', sep=',')
    node_data = node_data.set_index('Node')

    pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')  # rename node 'i' to 'Destest_Supply'

    simple_district = ug.UESGraph()
    simple_district.add_building(name="Destest_Supply", position=Point(44.0, -12.0), is_supply_heating=True)

    # Add building and network nodes from node_data to the uesgraph object
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

    # Add Diameter[m], Length[m], Insulation Thickness[m] and U-Value [W/mK] to edges
    for index, row in pipe_data.iterrows():
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['diameter'] = row['Inner Diameter [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['length'] = row['Length [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['dIns'] = row['Insulation Thickness [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]]['kIns'] = row['U-value [W/mK]']

    # write demand data to graphs building nodes. all Simple_Districts have the same demand at each time step
    # demands differ for every time step
    demand_data = pd.read_csv(
        'https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/'
        + 'Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv',
        sep=';',
        index_col=0)

    demand_data.columns = demand_data.columns.str.replace(' / W', '')  # rename demand
    demand = demand_data["SimpleDistrict_1"].values  # only demand for one District is taken (as they're all the same)
    demand = [round(x, 1) for x in demand]  # this demand is rounded to 1 digit for better readability

    # demand is written to every Simple district as
    for bldg in simple_district.nodelist_building:
        simple_district.nodes[bldg]['dT_design'] = params_dict['dT_design']  # part of .prepare_graph function?
        simple_district.nodes[bldg]['m_flo_bypass'] = params_dict['m_flo_bypass']  # not (yet) part of .prepare_graph function!
        if not simple_district.nodes[bldg]['is_supply_heating']:
            simple_district.nodes[bldg]['input_heat'] = demand
            simple_district.nodes[bldg]['max_demand_heating'] = max(demand)
        else:
            # do i need this setting? or is it overwritten anyways by the prepare graph function?
            simple_district.nodes[bldg]['T_supply'] = [params_dict['t_supply']] * int(52560)
            simple_district.nodes[bldg]['p_supply'] = [params_dict['p_supply']] * int(52560)

    # write general simulation data to graph, doesnt that happen with the .prepare_model function?
    simple_district.graph['network_type'] = 'heating'
    simple_district.graph['T_nominal'] = params_dict['t_nominal']  # needed?
    simple_district.graph['p_nominal'] = params_dict['p_nominal']  # needed?
    simple_district.graph['T_ground'] = [params_dict['t_ground']] * int(52560)  # ground temp for every 10min time step

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = str(edge[0]) + 'to' + str(edge[1])
        # simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1   # part of the prepare graph function
        simple_district.edges[edge[0], edge[1]]['fac'] = 1.0
        simple_district.edges[edge[0], edge[1]]['roughness'] = 2.5e-5  # Ref

    # write m_flow_nominal to the graphs edges with uesgraph function
    sysmod_utils.estimate_m_flow_nominal(graph=simple_district, dT_design=params_dict['dT_design'], network_type='heating')

    # print("####")
    # for edge in simple_district.edges:
    #     print("diameter: " + str(simple_district.edges[edge[0], edge[1]]["diameter"]))  # why print this?
    # print("####")

    # Plotting / Visualization with pipe diameters scaling
    vis = ug.Visuals(simple_district)
    vis.show_network(
        save_as="uesgraph_destest_16_selfsized_jonas.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100)

    # dir_model = os.path.join(os.path.dirname(__file__), 'model')
    dir_models_sciebo = os.path.join(dir_sciebo, 'models')
    if not os.path.exists(dir_models_sciebo):
        os.mkdir(dir_models_sciebo)

    if long_name:
        save_name = "Destest_Jonas__T_{:.0f}_{:.0f}_{:.0f}__dT_{:.0f}__p_{:.0f}_{:.0f}_{:.0f}__dTbldng_{:.0f}" \
                    "__Tsupbldng_{:.0f}__CoP_{:.0f}__Tcon_{:.0f}__Teva_{:.0f}__mBy_{:.0f}" \
            .format(params_dict['t_supply'] - 273.15, params_dict['t_nominal'] - 273.15,
                    params_dict['t_return'] - 273.15,
                    params_dict['dT_design'], params_dict['p_supply'] / 1e3, params_dict['p_nominal'] / 1e3,
                    params_dict['p_return'] / 1e3, params_dict['dT_building'],
                    params_dict['t_supply_building'] - 273.15,
                    params_dict['cop_nominal'] * 10, params_dict['t_con_nominal'] - 273.15,
                    params_dict['t_eva_nominal'] - 273.15,
                    params_dict['m_flo_bypass'] * 1e6)
        # {} are placeholders, :.0f rounds to 0 digits. Pressure is divided to show Unit in [kPa], Temperature in [°C]
        # often there are problems with Dymola when model names have different Symbols than Letters, Numbers and Underscores
        dir_model = os.path.join(dir_models_sciebo, save_name)
        counter = 1
        while os.path.exists(dir_model):
            counter += 1
            if counter < 3:
                save_name += "__V" + str(counter)
                dir_model = os.path.join(dir_models_sciebo, save_name)
            else:
                save_name = save_name[:-4]
                save_name += "__V" + str(counter)
                dir_model = os.path.join(dir_models_sciebo, save_name)
    else:
        save_name = "Destest_Jonas_{}".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        dir_model = os.path.join(dir_models_sciebo, save_name)

    # Copied and modified from e11
    simple_district = sysmod_utils.prepare_graph(
        graph=simple_district,
        T_supply=[params_dict['t_supply']] * int(52560),
        p_supply=params_dict['p_supply'],
        T_return=params_dict['t_return'],
        p_return=params_dict['p_return'],
        dT_design=params_dict['dT_design'],  # inside the supply station?
        # m_flow_nominal=m_flo_nom, # can also be set for each pipe by the .estimate_m_flow_nominal method
        # dp_nominal=50000,
        dT_building=params_dict['dT_building'],
        T_supply_building=params_dict['t_supply_building'],
        cop_nominal=params_dict['cop_nominal'],
        T_con_nominal=params_dict['t_con_nominal'],
        T_eva_nominal=params_dict['t_eva_nominal'],
        # dTEva_nominal=None,
        # dTCon_nominal=None
    )

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
        T_nominal=params_dict['t_nominal'],
        p_nominal=params_dict['p_nominal'],
        t_ground_prescribed=[params_dict['t_ground']] * int(52560)
    )

    if save_params_to_csv:
        path_to_csv = dir_model + "/" + save_name + "_overview.csv"

        # w = csv.writer(open(path_to_csv, "w"))
        # for key, val in params_dict.items():
        #     w.writerow([key, val])

        overview_df = pd.DataFrame.from_records(params_dict, index=range(1))
        overview_df.to_csv(path_to_csv)


if __name__ == '__main__':
    main()

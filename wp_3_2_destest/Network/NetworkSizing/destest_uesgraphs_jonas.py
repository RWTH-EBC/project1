# -*- coding: utf-8 -*-

import datetime
import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import os

from uesgraphs.uesmodels.utilities import utilities as utils
from uesgraphs.systemmodels import utilities as sysmod_utils


def main():
    """
    Defines a building node dictionary and adds it to the graph.
    Adds network nodes to the graph and creates a district heating network
    graph.
    Exported to a json
    """

    # Read node and pipe data

    # 16 Buildings
    node_data = pd.read_csv('Node data.csv', sep=',')
    node_data = node_data.set_index('Node')
    # imports cvs mit Node, X-Position [m],Y-Position [m],Peak power [kW]
    # SimpleDistrict_7, 80.0,   48.0,   19.347279296900002 -> 16 mal (Häuser)
    # a,    20.0,   72.0,   38.694558593800004  -> 9 mal (network nodes)

    pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    # Beginning Node, Ending Node, Length [m], Inner Diameter [m],
    # Insulation Thickness [m], Peak Load [kW], Total pressure loss [Pa/m], U-value [W/mK]

    # rename node 'i' to 'Destest_Supply' for usability
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')

    # create emtpy uesgraph object
    simple_district = ug.UESGraph()

    # Add supply exemplary as a single addition, same coordinates as in node_data
    supply_heating_1 = simple_district.add_building(
        name="Destest_Supply", position=Point(44.0, -12.0),
        is_supply_heating=True)

    # Add building and network nodes from node_data to the uesgraph object
    for node_name, values in node_data.iterrows():
        if node_name.startswith("Simple"):
            simple_district.add_building(
                name=node_name,
                position=Point(
                    values['X-Position [m]'], values['Y-Position [m]']),
                is_supply_heating=False)
        else:
            simple_district.add_network_node(
                'heating',
                name=node_name,
                position=Point(
                    values['X-Position [m]'], values['Y-Position [m]']),
                is_supply_heating=False)

    # Help dictionary for drawing the connections / edges 16 buildings
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

    # Adding the edges
    for key, values in connection_dict_heating_nodes.items():
        for value in values:
            simple_district.add_edge(
                simple_district.nodes_by_name[key],
                simple_district.nodes_by_name[value])

    # Add Diameter and Length information
    for index, row in pipe_data.iterrows():
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]][
                'diameter'] = row['Inner Diameter [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]][
                'length'] = row['Length [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]][
                'dIns'] = row['Insulation Thickness [m]']
        simple_district.edges[
            simple_district.nodes_by_name[row['Beginning Node']],
            simple_district.nodes_by_name[row['Ending Node']]][
                'kIns'] = row['U-value [W/mK]']

    # Plotting / Visualization with pipe diameters scaling
    vis = ug.Visuals(simple_district)
    vis.show_network(
        save_as="uesgraph_destest_16.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100
    )

    # write demand data to graph

    demand_data = pd.read_csv(
        'https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv',
        sep=';',
        index_col=0)

    demand_data.columns = demand_data.columns.str.replace(' / W', '')

    demand = demand_data["SimpleDistrict_1"].values
    demand = [round(x, 1) for x in demand]

    for bldg in simple_district.nodelist_building:
        if not simple_district.nodes[bldg]['is_supply_heating']:
            simple_district.nodes[bldg]['input_heat'] = demand
            simple_district.nodes[bldg]['max_demand_heating'] = max(demand)
        else:
            simple_district.nodes[bldg]['T_supply'] = [273.15 + 50]
            simple_district.nodes[bldg]['p_supply'] = [3.4e5]

    # write general simulation data to graph
    # values needs to be revised for common excersise

    end_time = 365 * 24 * 3600
    time_step = 600
    n_steps = end_time / time_step

    simple_district.graph['network_type'] = 'heating'
    simple_district.graph['T_nominal'] = 273.15 + 50
    simple_district.graph['p_nominal'] = 3e5
    simple_district.graph['T_ground'] = [285.15] * int(n_steps)

    for node in simple_district.nodelist_building:
        simple_district.nodes[node]['dT_design'] = 20
        simple_district.nodes[node]['m_flo_bypass'] = 0.5  #wird dem mako tempalte übergeben um personalisierte .mo files zu schreiben

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = \
            str(edge[0]) + 'to' + str(edge[1])
        simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1
        simple_district.edges[edge[0], edge[1]]['fac'] = 1.0
        simple_district.edges[edge[0], edge[1]]['roughness'] = 2.5e-5   # Ref

    print("####")

    # special m_flow estimation
    simple_district = sysmod_utils.estimate_m_flow_nominal_tablebased(
        simple_district,
        network_type='heating')

    # peak power m_flow estimation
    simple_district = sysmod_utils.estimate_m_flow_nominal(
        simple_district,
        dT_design=20,
        network_type='heating')

    dir_model = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)

    for edge in simple_district.edges:
        print(simple_district.edges[edge[0], edge[1]]["diameter"])

    for edge in simple_district.edges:
        simple_district.edges[edge[0], edge[1]]["diameter"] = 0

    print("####")

    simple_district = utils.size_hydronic_network(
        graph=simple_district,
        network_type="heating",
        delta_t_heating=20,
        dp_set=100.0,
        loop=False)

    # Plotting / Visualization with pipe diameters scaling
    vis = ug.Visuals(simple_district)
    vis.show_network(
        save_as="uesgraph_destest_16_selfsized_jonas.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100
    )

    for edge in simple_district.edges:
        print(simple_district.edges[edge[0], edge[1]]["diameter"])

    #### Copied and modified from e11 ####
    # To add data for model generation to the uesgraph the prepare_graph
    # function is used. There are thirteen parameters available. Below the supply
    # temperature in K, supply pressure in Pa, return temperature in K,
    # return pressure in Pa, Design temperature difference over substation in K
    # and the nominal mass flow rate in kg/s are added to the graph.
    simple_district = sysmod_utils.prepare_graph(
        graph=simple_district,
        T_supply=[273.15 + 90],
        p_supply=13e5,
        T_return=273.15 + 45,   #funktion aus t_supply ind dT_design -> könnte man auch weglassen?
        p_return=2e5,
        dT_design=30,
        m_flow_nominal=1,
    )

    # ---Copied and modified from e11---
    # To generate a generic Modelica model the create_model function is used.
    # There are 21 parameters available.
    sysmod_utils.create_model(
        name="Destest_Jonas_von_E11_Pinola_test_bypass_45_70_90",
        save_at=dir_model,
        graph=simple_district,
        stop_time=end_time,
        timestep=time_step,
        model_supply='AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdeal',
        model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        # model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDp',    # aus E11
        model_pipe="AixLib.Fluid.FixedResistances.PlugFlowPipe",
        model_medium="AixLib.Media.Specialized.Water.ConstantProperties_pT",
        model_ground="t_ground_table",
        T_nominal=273.15 + 70,
        p_nominal=3e5,
    )


# Main function
if __name__ == '__main__':
    main()
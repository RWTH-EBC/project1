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
    graph. Creates a Modelica Model of the Graph.
    """

    # Read node and pipe data for 16 buildings
    node_data = pd.read_csv('Node data.csv', sep=',')
    node_data = node_data.set_index('Node')
    # imports cvs with Node, X-Position [m],Y-Position [m],Peak power [kW]
    # SimpleDistrict_7, 80.0,   48.0,   19.347279296900002 -> 16 times (building nodes)
    # a,    20.0,   72.0,   38.694558593800004  -> 9 times (network nodes)

    pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    # Beginning Node, Ending Node, Length [m], Inner Diameter [m],
    # Insulation Thickness [m], Peak Load [kW], Total pressure loss [Pa/m], U-value [W/mK]
    # there are 24 pipes for th 16 building network.

    # rename node 'i' to 'Destest_Supply' for better usability
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')

    # create emtpy uesgraph object
    simple_district = ug.UESGraph()

    # Add supply exemplary as a single addition, same coordinates as in node_data
    # would be more consistent if the coordinates would be taken right out of the node_data.csv!
    supply_heating_1 = simple_district.add_building(name="Destest_Supply", position=Point(44.0, -12.0),
                                                    is_supply_heating=True)

    # Add building and network nodes from node_data to the uesgraph object
    for node_name, values in node_data.iterrows():
        if node_name.startswith("Simple"):  # all Simple_districs are added as buildings
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
            simple_district.add_edge(
                simple_district.nodes_by_name[key],
                simple_district.nodes_by_name[value])

    # Add Diameter and Length information to the edges from pipe_data.csv
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
    # not needed for model creation, just for checking
    vis = ug.Visuals(simple_district)
    vis.show_network(
        save_as="uesgraph_destest_16.png",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100
    )

    # write demand data to graphs building nodes. all Simple_Districts have the same demand at each time step
    # demands differ for every time step
    demand_data = pd.read_csv(
        'https://raw.githubusercontent.com/ibpsa/project1/master/wp_3_1_destest/'
        + 'Buildings/SimpleDistrict/Results/SimpleDistrict_IDEAS/SimpleDistrict_district.csv',
        sep=';',
        index_col=0)

    demand_data.columns = demand_data.columns.str.replace(' / W', '')   # rename demand
    demand = demand_data["SimpleDistrict_1"].values  # only demand for one District is taken (as they're all the same)
    demand = [round(x, 1) for x in demand]  # this demand is rounded to 1 digit for better readability

    m_flo_bypass = 0.005  # set as variable for dynamic model naming
    # demand is written to every Simple district as
    for bldg in simple_district.nodelist_building:
        simple_district.nodes[bldg]['dT_design'] = 20  # part of .prepare_graph function
        simple_district.nodes[bldg]['m_flo_bypass'] = m_flo_bypass  # not (yet) part of .prepare_graph function!
        if not simple_district.nodes[bldg]['is_supply_heating']:
            simple_district.nodes[bldg]['input_heat'] = demand
            simple_district.nodes[bldg]['max_demand_heating'] = max(demand)
       # else:
            #do i need this setting? or is it overwritten anyways by the prepare graph function?
            # simple_district.nodes[bldg]['T_supply'] = [273.15 + 50]
            # simple_district.nodes[bldg]['p_supply'] = [3.4e5]

    # write general simulation data to graph, doesnt that happen with the .prepare_model function?
    simple_district.graph['network_type'] = 'heating'
    # simple_district.graph['T_nominal'] = 273.15 + 50   # needed?
    # simple_district.graph['p_nominal'] = 3e5   # needed?
    simple_district.graph['T_ground'] = [285.15] * int(52560) # sets ground temp for every 10min time step.

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = \
            str(edge[0]) + 'to' + str(edge[1])
        simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1   # part of the prepare graph function
        simple_district.edges[edge[0], edge[1]]['fac'] = 1.0
        simple_district.edges[edge[0], edge[1]]['roughness'] = 2.5e-5  # Ref

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
        print("diameter: " + str(simple_district.edges[edge[0], edge[1]]["diameter"]))  # why print this?

    print("####")

    # what for?
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
        scaling_factor_diameter=100)

    # .prepare_graph() variables, for dynamic model naming
    t_sup = [273.15 + 87]   # has to be a list object to work properly with the .prepare_graph function
    p_sup = 13e5
    t_ret = 273.15 + 57  # function of T_supply and dT_design?
    p_ret = 2e5
    dt_des = 30
    m_flo_nom = 1

    # .create_model() variables, for dynamic model naming
    t_nom = 273.15 + 30  # equals T_Ambient in Dymola? Start value for every pipe?
    p_nom = 3e5

    # Copied and modified from e11
    simple_district = sysmod_utils.prepare_graph(
        graph=simple_district,
        T_supply=t_sup,
        p_supply=p_sup,
        T_return=t_ret,  # function aus t_supply ind dT_design? -> redundant?
        p_return=p_ret,
        dT_design=dt_des,
        m_flow_nominal=m_flo_nom,
    )

    # To generate a generic Modelica model the create_model function is used. There are 21 parameters available.
    sysmod_utils.create_model(
        name="Destest_Jonas__T_{:.0f}_{:.0f}_{:.0f}__dT_{}__p_{:.0f}_{:.0f}_{:.0f}__mNom_{:.0f}__mByGram_{:.0f}"
            .format(t_sup[0] - 273.15, t_nom - 273.15, t_ret - 273.15, dt_des,
                    p_sup / 1e5, p_nom / 1e5, p_ret / 1e5, m_flo_nom, m_flo_bypass * 1e3),
        # {} are placeholders, :.0f rounds to 0 digits. Pressure is divided to show Unit in [bar], Temperature in [°C]
        # often there are problems with Dymola when model names
        # have different Symbols than Letters, Numbers and Underscores
        save_at=dir_model,
        graph=simple_district,
        stop_time=365 * 24 * 3600,
        timestep=600,
        model_supply='AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdeal',
        # model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDp',  # aus E11
        model_pipe="AixLib.Fluid.FixedResistances.PlugFlowPipe",
        model_medium="AixLib.Media.Specialized.Water.ConstantProperties_pT",
        model_ground="t_ground_table",
        T_nominal=t_nom,
        p_nominal=p_nom,
    )


if __name__ == '__main__':
    main()

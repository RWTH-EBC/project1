# -*- coding: utf-8 -*-

import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import os
from uesgraphs.systemmodels import utilities as sysmod_utils


def main():
    """
    Defines a building node dictionary and adds it to the graph.
    Adds network nodes to the graph and creates a district heating network
    graph. Creates a Modelica Model of the Graph.
    """

    # Parameters
    t_sup = 273.15 + 86
    t_nom = 273.15 + 60  # equals T_Ambient in Dymola? Start value for every pipe?
    dt_des = 30
    t_ret = t_sup - dt_des  # function of T_supply and dT_design?
    t_grnd = 273.15 + 10

    p_sup = 4e5
    p_nom = 3e5
    p_ret = 2e5

    m_flo_nom = 1
    m_flo_bypass = 0.5

    node_data = pd.read_csv('Node_data_8_buildings.csv', sep=',')
    node_data = node_data.set_index('Node')

    pipe_data = pd.read_csv('Pipe_data_8_buildings.csv', sep=',')
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')  # rename node 'i' to 'Destest_Supply'

    simple_district = ug.UESGraph()
    simple_district.add_building(name="Destest_Supply", position=Point(44.0, -12.0), is_supply_heating=True)

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

    # Help dictionary for drawing the connections / edges 8 buildings
    connection_dict_heating_nodes = {
        "c": ["d", "SimpleDistrict_10", "SimpleDistrict_11"],
        "g": ["h", "SimpleDistrict_9", "SimpleDistrict_12"],
        "d": ["Destest_Supply", "SimpleDistrict_16", "SimpleDistrict_15"],
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

    # demand is written to every Simple district as
    for bldg in simple_district.nodelist_building:
        simple_district.nodes[bldg]['dT_design'] = dt_des  # part of .prepare_graph function?
        simple_district.nodes[bldg]['m_flo_bypass'] = m_flo_bypass  # not (yet) part of .prepare_graph function!
        if not simple_district.nodes[bldg]['is_supply_heating']:
            simple_district.nodes[bldg]['input_heat'] = demand
            simple_district.nodes[bldg]['max_demand_heating'] = max(demand)
        else:
            # do i need this setting? or is it overwritten anyways by the prepare graph function?
            simple_district.nodes[bldg]['T_supply'] = [t_sup]
            simple_district.nodes[bldg]['p_supply'] = [p_sup]

    # write general simulation data to graph, doesnt that happen with the .prepare_model function?
    simple_district.graph['network_type'] = 'heating'
    simple_district.graph['T_nominal'] = t_nom   # needed?
    simple_district.graph['p_nominal'] = p_nom   # needed?
    simple_district.graph['T_ground'] = [t_grnd] * int(52560)   # sets ground temp for every 10min time step.

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = str(edge[0]) + 'to' + str(edge[1])
        simple_district.edges[edge[0], edge[1]]['fac'] = 1.0
        simple_district.edges[edge[0], edge[1]]['roughness'] = 2.5e-5  # Ref

    print("####")

    dir_model = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)

    for edge in simple_district.edges:
        print("diameter: " + str(simple_district.edges[edge[0], edge[1]]["diameter"]))  # why print this?

    print("####")

    # Plotting / Visualization with pipe diameters scaling
    vis = ug.Visuals(simple_district)
    vis.show_network(
        save_as="uesgraph_destest_8_selfsized_jonas.png", show_diameters=True, scaling_factor=15,
        labels="name", label_size=10, scaling_factor_diameter=100)

    save_name = "Destest_8_Jonas__T_{:.0f}_{:.0f}_{:.0f}__dT_{}__p_{:.0f}_{:.0f}_{:.0f}__mNom_{:.0f}__mByGram_{:.0f}"\
        .format(t_sup - 273.15, t_nom - 273.15, t_ret - 273.15, dt_des,
                p_sup / 1e5, p_nom / 1e5, p_ret / 1e5, m_flo_nom, m_flo_bypass * 1e3)

    simple_district = sysmod_utils.prepare_graph(
        graph=simple_district,
        T_supply=[t_sup] * int(52560),
        p_supply=p_sup,
        T_return=t_ret,  # function aus t_supply ind dT_design? -> redundant?
        p_return=p_ret,
        dT_design=dt_des,   # inside the supply station?
        m_flow_nominal=m_flo_nom,
        # dp_nominal=None,
        # dT_building=None,   # inside the buildings/demand stations?
        # T_supply_building=None,
        # cop_nominal=None,
        # T_con_nominal=None,
        # T_eva_nominal=None,
        # dTEva_nominal=None,
        # dTCon_nominal=None
    )

    # To generate a generic Modelica model the create_model function is used. There are 21 parameters available.
    sysmod_utils.create_model(
        name=save_name,
        save_at=dir_model,
        graph=simple_district,
        stop_time=365 * 24 * 3600,
        timestep=600,
        model_supply='AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdeal',
        # model_supply='AixLib.Fluid.DistrictHeatingCooling.Supplies.OpenLoop.SourceIdealPump',
        model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDpFixedTempDifferenceBypass',
        # model_demand='AixLib.Fluid.DistrictHeatingCooling.Demands.OpenLoop.VarTSupplyDp',  # aus E11
        model_pipe="AixLib.Fluid.FixedResistances.PlugFlowPipe",
        model_medium="AixLib.Media.Specialized.Water.ConstantProperties_pT",
        model_ground="t_ground_table",
        T_nominal=t_nom,
        p_nominal=p_nom,
        t_ground_prescribed=[t_grnd] * int(52560)
    )


if __name__ == '__main__':
    main()
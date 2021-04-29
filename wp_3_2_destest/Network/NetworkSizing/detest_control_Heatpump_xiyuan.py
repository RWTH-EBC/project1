# -*- coding: utf-8 -*-

import datetime
import uesgraphs as ug
from shapely.geometry import Point
import pandas as pd
import os

# from uesmodels.uesmodel import UESModel
# import uesmodels as um

from uesgraphs.systemmodels import systemmodelheating as sysmh
from uesgraphs.systemmodels import utilities as sysmod_utils


def main():
    """
    Defines a building node dictionary and adds it to the graph.
    Adds network nodes to the graph and creates a district heating network
    graph.
    Exported to a json
    """

    # Read node and pipe data
    node_data = pd.read_csv(
        'https://raw.githubusercontent.com/ibpsa/project1/WP3/'
        'wp_3_2_destest/Network/NetworkSizing/Node%20data.csv', sep=',')

    pipe_data = pd.read_csv('Pipe_data.csv', sep=',')
    # node_data = pd.read_csv('Node data.csv', sep=',')

    node_data = node_data.set_index('Node')

    # renaming for usability
    pipe_data = pipe_data.replace(to_replace='i', value='Destest_Supply')

    simple_district = ug.UESGraph()

    # Add supply exemplary as a single addition
    supply_heating_1 = simple_district.add_building(
        name="Destest_Supply", position=Point(44.0, -12.0),
        is_supply_heating=True)

    # Add building and network nodes
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

    # Help dictionary for drawing the connections / edges
    connection_dict_heating_nodes = {
        "a": ["b", "SimpleDistrict_2", "SimpleDistrict_3"],
        "b": ["c", "SimpleDistrict_5", "SimpleDistrict_6"],
        "c": ["d", "SimpleDistrict_10", "SimpleDistrict_11"],
        "e": ["f", "SimpleDistrict_1", "SimpleDistrict_4"],
        "f": ["g", "SimpleDistrict_8", "SimpleDistrict_7"],
        "g": ["h", "SimpleDistrict_9", "SimpleDistrict_12"],
        "d": ["Destest_Supply", "SimpleDistrict_16", "SimpleDistrict_15"],
        "h": ["Destest_Supply", "SimpleDistrict_14", "SimpleDistrict_13"]}

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
        save_as="uesgraph_destest.pdf",
        show_diameters=True,
        scaling_factor=15,
        labels="name",
        label_size=10,
        scaling_factor_diameter=100
    )

    # write demand data to graph, positive is heating, negative is cooling

    dem_path = "/Users/jonasgrossmann/sciebo/RWTH_Dokumente/MA_Masterarbeit_RWTH" \
               "/Data/demand_profiles/demand_data_MA_Xiyuan.xlsx"
    demand_data = pd.read_excel(dem_path, sheet_name='Tabelle4', sep =',')
    DHW_data = pd.read_excel(dem_path, sheet_name='Tabelle3', sep =',')

    # x = range(0, 31536000, 900)
    # index_to_drop = []
    # j = 1
    # print('starting to iterate over df')
    # for i, row in enumerate(demand_data.iterrows()):
    #     if demand_data.index[i] not in x:
    #         index_to_drop.append(demand_data.index[i])
    # demand_data = demand_data.drop(index_to_drop)

    demand_data.columns = demand_data.columns.str.replace(' / W', '')

    for bldg in simple_district.nodelist_building:
        if not simple_district.nodes[bldg]['is_supply_heating']:
            demand = demand_data[
                simple_district.nodes[bldg]['name']].values.tolist()
            demand = demand_data[
                simple_district.nodes[bldg]['name']].values
            demand = [round(x, 1) for x in demand]
            demand1 = DHW_data[
                simple_district.nodes[bldg]['name']].values.tolist()
            demand1 = DHW_data[
                simple_district.nodes[bldg]['name']].values
            demand1 = [round(x,1) for x in demand1]
            # demand = [x if x else 83.6 for x in demand]
            simple_district.nodes[bldg]['input_heat'] = demand
            demand_all = [demand, demand1]
            simple_district.nodes[bldg]['input_all'] = []
            for i in range(len(demand)):
                simple_district.nodes[bldg]['input_all'].append([row[i] for row in demand_all])
        else:

            JAN = [273.15+10]*4464
            FEB = [273.15+10]*4032
            MAR = [273.15+10]*4464
            APR = [273.15+10]*4320
            MAY = [273.15+10]*4464
            JUN = [273.15+10]*4320
            JUL = [273.15+10]*4464
            AUG = [273.15+10]*4464
            SEP = [273.15+10]*4320
            OCT = [273.15+10]*4464
            NOV = [273.15+10]*4320
            DEC = [273.15+10]*4465
            simple_district.nodes[bldg]['TIn'] = JAN+FEB+MAR+APR+MAY+JUN+JUL+AUG+SEP+OCT+NOV+DEC
            
            
            simple_district.nodes[bldg]['dpIn'] = [5e5]

    # write general simulation data to graph
    # values needs to be revised for common excersise

    end_time = 365 * 24 * 3600
    time_step = 600
    n_steps = end_time / time_step

    simple_district.graph['network_type'] = 'heating'
    simple_district.graph['T_nominal'] = 273.15 + 10
    simple_district.graph['p_nominal'] = 3e5
    JAN_G = [273.15+0]*4464
    FEB_G = [273.15+0]*4032
    MAR_G = [273.15+4]*4464
    APR_G = [273.15+10]*4320
    MAY_G = [273.15+16]*4464
    JUN_G = [273.15+20.5]*4320
    JUL_G = [273.15+23]*4464
    AUG_G = [273.15+22]*4464
    SEP_G = [273.15+16]*4320
    OCT_G = [273.15+10]*4464
    NOV_G = [273.15+4]*4320
    DEC_G = [273.15+1]*4465
    simple_district.graph['T_ground'] = JAN_G+ FEB_G+ MAR_G+ APR_G+ MAY_G+ JUN_G+ JUL_G+ AUG_G+ SEP_G+ OCT_G+ NOV_G+ DEC_G

    for node in simple_district.nodelist_building:
        simple_district.nodes[node]['dT_design'] = 5
        simple_district.nodes[node]['T_return'] = 273.15 + 20

    for edge in simple_district.edges():
        simple_district.edges[edge[0], edge[1]]['name'] = \
            str(edge[0]) + 'to' + str(edge[1])
        simple_district.edges[edge[0], edge[1]]['m_flow_nominal'] = 1
        simple_district.edges[edge[0], edge[1]]['fac'] = 1.0
        simple_district.edges[edge[0], edge[1]]['roughness'] = 2.5e-5   # Ref

    # special m_flow estimation
    # simple_district = um.utilities.estimate_m_flow_nominal_tablebased(
    #     simple_district,
    #     network_type='heating')

    # peak power m_flow estimation
    # simple_district = um.utilities.estimate_m_flow_nominal(
    #     simple_district,
    #     dT_design=20,
    #     network_type='heating')
    sysmod_utils.estimate_m_flow_nominal(graph=simple_district, dT_design=20,
                                         network_type='heating', input_heat_str='input_heat')

    dir_model = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)

    new_model = sysmh.SystemModelHeating(network_type=simple_district.graph["network_type"])
    new_model.stop_time = end_time
    new_model.timestep = time_step
    new_model.tolerance = 1e-5
    new_model.T_nominal = 273.15 + 10
    new_model.fraction_glycol = 0.6
    new_model.p_nominal = 3e5
    new_model.import_from_uesgraph(simple_district)
    new_model.set_connection(remove_network_nodes=True)
    new_model.add_ground_around_pipe = False
    new_model.ground_buried_cylindric = False
    new_model.ground_model = 't_ground_table'
    new_model.with_heat_flow_output = True
    new_model.with_heat_loss_output = True

    new_model.set_control_pressure(
        name_building='max_distance',
        #name_building='T1010_W',
        dp=1.2e5,
        name_supply='Destest_Supply',
        p_max=13e5
    )

    package = 'AixLib.Fluid.DistrictHeatingCooling.'
    model_supply_ideal = package + 'Supplies.ClosedLoop.IdealPlantPump'    # Supplies.ClosedLoop.IdealSourcewithT_supply
#    model_supply_power = package + 'Supplies.OpenLoop.SourcePowerDoubleMvar'
    model_demand = package +\
        'Demands.ClosedLoop.PumpControlledwithHP_V4'
    model_pipe = 'AixLib.Fluid.FixedResistances.PlugFlowPipe'

    new_model.add_return_network = True
    new_model.medium = 'AixLib.Media.Antifreeze.PropyleneGlycolWater'
#    new_model.mediumbuilding = 'Aixlib_Media_Water'
    new_model.p_nominal = 3e5

    for node in new_model.nodelist_building:
        is_supply = 'is_supply_{}'.format(new_model.network_type)
        if new_model.nodes[node][is_supply]:
            new_model.nodes[node]['comp_model'] = model_supply_ideal
            new_model.nodes[node]['dp_nominal'] = 3000
        else:
            new_model.nodes[node]['comp_model'] = model_demand
            input_heat = new_model.nodes[node]['input_heat']
            if new_model.network_type == 'heating':
                new_model.nodes[node]['heatDemand_max'] = max(input_heat)
               #new_model.nodes[node]['coolingDemand_max'] = -5000
                new_model.nodes[node]['T_supplyHeating'] = 273.15+35
                new_model.nodes[node]['T_supplyCooling']= 273.15+5

    for node in new_model.nodelist_pipe:
        new_model.nodes[node]['comp_model'] = model_pipe

    simulation_setup = {'data': {'start_time': 0,
                                 'stop_time': new_model.stop_time,
                                 'output_interval': new_model.timestep},
                        }

    print('dir_model', dir_model)
    time_stamp = int(round(datetime.datetime.now().timestamp(), 0))
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Write Modelica code
    new_model.model_name = 'Sim' + str(time_stamp) + "Destest" +\
        '_clossed_loop_Tvor'
    new_model.write_modelica_package(save_at=dir_model)

    
# Main function
if __name__ == '__main__':
    main()

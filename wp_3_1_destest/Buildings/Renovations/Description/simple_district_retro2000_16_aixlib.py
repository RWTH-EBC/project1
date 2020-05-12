# -*- coding: utf-8 -*-
# @Author: MichaMans
# @Date:   2018-09-20 11:05:56
# @Last Modified by:   MichaMans
# @Last Modified time: 2019-03-21 18:38:09

"""This module contains an example how to generate archetype buildings using
TEASER API functions. THis script currently only works with TEASER branch
issue539_setpoints
"""
import os
from teaser.project import Project
from dymola.dymola_interface import DymolaInterface
import pandas as pd
from teaser.logic.buildingobjects.buildingphysics.layer import Layer
from teaser.logic.buildingobjects.buildingphysics.material import Material
import simulate as sim


def example_generate_simple_district_building(prj, nr_of_bldg):
    """"This function demonstrates the generation of residential and
    non-residential archetype buildings using the API function of TEASER"""

    """First step: Import the TEASER API (called Project) into your Python module
    To use the API instantiate the Project class and rename the Project. The
    parameter load_data=True indicates that we load `iwu` typology archetype
    data into our Project (e.g. for Material properties and typical wall
    constructions. This can take a few seconds, depending on the size of the
    used data base). Be careful: Dymola does not like whitespaces in names and
    filenames, thus we will delete them anyway in TEASER."""

    # There are two different types of archetype groups: residential and
    # non-residential buildings. Two API functions offer the opportunity to
    # generate specific archetypes.

    """To generate residential archetype buildings the function
    Project.add_residential() is used. Seven parameters are compulsory,
    additional parameters can be set according to the used method. `method`
    and `usage` are used to distinguish between different archetype
    methods. The name, year_of_construction, number and height of floors
    and net_leased_area need to be set to provide enough information for
    archetype generation. For specific information on the parameters please
    read the docs."""

    for bldg_number in range(1, nr_of_bldg + 1):

        bldg = prj.add_residential(
            method="tabula_de",
            usage="single_family_house",
            name="SimpleDistrictBuilding_2000_{}".format(bldg_number),
            year_of_construction=1980,
            number_of_floors=2,
            height_of_floors=3.5,
            net_leased_area=128,
            construction_type="tabula_standard",
        )

        bldg.zone_area_factors = {
            "SingleDwelling": [0.5, "Living"],
            "BedRoom": [0.5, "Bed room"],
        }

        bldg.generate_archetype()

        layers_ow = [
            ["HeavyMasonryForExteriorApplications", 0.1, 1850, 1.1, 0.84, 0.55, 0.9],
            ["LargeCavityHorizontalHeatTransfer", 0.1, 100, 0.5555, 0.02, 0.55, 0.9],
            ["Rockwool", 0.07, 110, 0.036, 0.84, 0.8, 0.9],
            ["MediumMasonryForExteriorApplications", 0.14, 1400, 0.75, 0.84, 0.55, 0.9],
            ["GypsumPlasterForFinishing", 0.02, 975, 0.6, 0.84, 0.65, 0.9],
        ]

        layers_iw = [
            ["GypsumPlasterForFinishing", 0.02, 975, 0.6, 0.84, 0.65, 0.9],
            ["MediumMasonryForInteriorApplications", 0.14, 1400, 0.54, 0.84, 0.55, 0.9],
            ["GypsumPlasterForFinishing", 0.02, 975, 0.6, 0.84, 0.65, 0.9],
        ]

        layers_dz_gf = [
            ["DenseCastConcreteAlsoForFinishing", 0.15, 2100, 1.4, 0.84, 0.55, 0.9],
            ["ExpandedPolystrenemOrEPS", 0.09, 26, 0.036, 1.47, 0.8, 0.9],
            ["ScreedOrLightCastConcrete", 0.08, 1100, 0.6, 0.84, 0.55, 0.9],
            ["CeramicTileForFinishing", 0.02, 2100, 1.4, 0.84, 0.55, 0.9],
        ]

        layers_dz_ceiling = [
            ["DenseCastConcreteAlsoForFinishing", 0.1, 2100, 1.4, 0.84, 0.55, 0.9],
            ["GypsumPlasterForFinishing", 0.02, 975, 0.6, 0.84, 0.65, 0.9],
        ]

        layers_dz_floor = [
            ["TimberForFinishing", 0.02, 550, 0.11, 1.88, 0.44, 0.9],
            ["ExpandedPolystrenemOrEPS", 0.08, 1100, 0.6, 0.84, 0.55, 0.9],
            ["ScreedOrLightCastConcrete", 0.1, 2100, 1.4, 0.84, 0.55, 0.9],
        ]

        layers_nz_rt = [
            ["CeramicTileForFinishing", 0.025, 2100, 1.4, 0.84, 0.55, 0.9],
            ["LargeCavityVerticalHeatTransfer", 0.1, 100, 0.625, 0.02, 0.85, 0.9],
            ["Glasswool", 0.12, 80, 0.04, 0.84, 0.85, 0.9],
            ["GypsumPlasterForFinishing", 0.02, 975, 0.6, 0.84, 0.65, 0.9],
        ]

        for zone in bldg.thermal_zones:
            # outer walls and windows equal for every zone
            for wall in zone.outer_walls:
                wall.area = 22.4
                wall.layer = None
                for lay in layers_ow:
                    temp_layer = Layer(parent=wall)
                    temp_layer.thickness = lay[1]
                    temp_layer_material = Material(parent=temp_layer)
                    temp_layer_material.name = lay[0]
                    temp_layer_material.density = lay[2]
                    temp_layer_material.thermal_conduc = lay[3]
                    temp_layer_material.heat_capac = lay[4]
                    temp_layer_material.solar_absorp = lay[5]
                    temp_layer_material.ir_emissivity = lay[6]

            for win in zone.windows:
                win.area = 5.6
                win.layer = None
                # total u-value = 0.15*2.6+0.75*1.4 ~ 1.58
                # equivalent thickness with lamda=0.76, alpha out = 25 alpha in = 7.7
                # d_equivalent = 0.3519
                win.g_value = 0.755
                win.a_conv = 0.02
                temp_layer = Layer(parent=win)
                temp_layer.thickness = 0.3519
                temp_layer_material = Material(parent=temp_layer)
                temp_layer_material.name = "Glas_equivalent_lamda0.76"
                temp_layer_material.density = 1
                temp_layer_material.thermal_conduc = 0.76
                temp_layer_material.heat_capac = 1
                temp_layer_material.solar_absorp = 0.7
                temp_layer_material.ir_emissivity = 0.9

            if zone.name == "SingleDwelling":
                zone.rooftops = None

                for gf in zone.ground_floors:
                    gf.area = 64
                    gf.layer = None
                    for lay in layers_dz_gf:
                        temp_layer = Layer(parent=gf)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]
                for ceiling in zone.ceilings:
                    ceiling.area = 64
                    ceiling.layer = None
                    for lay in layers_dz_ceiling:
                        temp_layer = Layer(parent=ceiling)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]
                for floor in zone.floors:
                    floor.area = 64
                    floor.layer = None
                    for lay in layers_dz_floor:
                        temp_layer = Layer(parent=floor)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]
                for iw in zone.inner_walls:
                    iw.area = 187
                    iw.layer = None
                    for lay in layers_iw:
                        temp_layer = Layer(parent=iw)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]

            if zone.name == "BedRoom":
                zone.rooftops.pop(0)
                zone.ceiling = None
                zone.floors = None
                zone.ground_floors = None
                for rt in zone.rooftops:
                    rt.area = 64
                    rt.layer = None
                    rt.tilt = 0
                    rt.orientation = -1
                    for lay in layers_nz_rt:
                        temp_layer = Layer(parent=rt)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]
                for iw in zone.inner_walls:
                    iw.area = 168
                    iw.layer = None
                    for lay in layers_iw:
                        temp_layer = Layer(parent=iw)
                        temp_layer.thickness = lay[1]
                        temp_layer_material = Material(parent=temp_layer)
                        temp_layer_material.name = lay[0]
                        temp_layer_material.density = lay[2]
                        temp_layer_material.thermal_conduc = lay[3]
                        temp_layer_material.heat_capac = lay[4]
                        temp_layer_material.solar_absorp = lay[5]
                        temp_layer_material.ir_emissivity = lay[6]

    prj.calc_all_buildings()

    profile_living = [
        291.15,
        291.15,
        291.15,
        291.15,
        291.15,
        291.15,
        291.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        294.15,
        294.15,
        294.15,
        294.15,
        294.15,
        294.15,
        291.15,
    ]

    profile_bed_room = [
        293.15,
        293.15,
        293.15,
        293.15,
        293.15,
        293.15,
        293.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        289.15,
        291.15,
        291.15,
        291.15,
        291.15,
        291.15,
        291.15,
        293.15,
    ]

    for bldg in prj.buildings:
        bldg.thermal_zones[0].use_conditions.heating_profile = profile_living

        bldg.thermal_zones[1].use_conditions.heating_profile = profile_bed_room

        # total max power of in QNight_zone = 6 W/m² * 64 m² = 384 W
        # number of max persons for nightzone 3.84

        # Update TEASER version 0.7.3
        # persons : float [Persons/m2]
        #    Specific number of persons per square area.
        # Internal Gains are modelled as persons only
        # Maximum QDay_max = 1280 W
        # specific power = 20 W/m²
        # specific Power person = 100 W/Pers
        # equals specific number of Persion of 20 / 100 = 0.2 Pers/m²
        # This seems to be a very high value - Check where this values comes from

        bldg.thermal_zones[0].use_conditions.persons = 0.2
        bldg.thermal_zones[0].use_conditions.fixed_heat_flow_rate_persons = 100
        bldg.thermal_zones[0].use_conditions.ratio_conv_rad_persons = 0.5

        bldg.thermal_zones[0].use_conditions.machines = 0
        bldg.thermal_zones[0].use_conditions.lighting_power = 0
        bldg.thermal_zones[0].use_conditions.infiltration_rate = 0.2
        bldg.thermal_zones[0].use_conditions.use_constant_infiltration = True

        # Update TEASER version 0.7.3
        # persons : float [Persons/m2]
        #    Specific number of persons per square area.
        # Internal Gains are modelled as persons only
        # Maximum QDay_max = 384 W
        # specific power = 6 W/m²
        # specific Power person = 100 W/Pers
        # equals specific number of Persion of 6 / 100 = 0.06 Pers/m²

        bldg.thermal_zones[1].use_conditions.persons = 0.06
        bldg.thermal_zones[1].use_conditions.fixed_heat_flow_rate_persons = 100
        bldg.thermal_zones[1].use_conditions.ratio_conv_rad_persons = 0.5

        bldg.thermal_zones[1].use_conditions.machines = 0
        bldg.thermal_zones[1].use_conditions.lighting_power = 0
        bldg.thermal_zones[1].use_conditions.infiltration_rate = 0.2
        bldg.thermal_zones[1].use_conditions.use_constant_infiltration = True

        # profiles for day and night zone representing the share of total number
        # of persons

        bldg.thermal_zones[0].use_conditions.persons_profile = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]

        bldg.thermal_zones[1].use_conditions.persons_profile = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.21,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
        ]

        bldg.thermal_zones[0].model_attr.heat_load = 8024.46
        bldg.thermal_zones[1].model_attr.heat_load = 8548.50

    return prj


if __name__ == "__main__":

    prj = Project(load_data=True)
    prj.name = "Simple_District_Retrofit2000_Destest_AixLib"
    prj.used_library_calc = "AixLib"
    prj.number_of_elements_calc = 2
    prj.weather_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "SimpleDistrict",
        "Climate",
        "BEL_Brussels.064510_IWEC.mos",
    )

    prj = example_generate_simple_district_building(prj=prj, nr_of_bldg=1)
    # To make sure the parameters are calculated correctly we recommend to
    # run calc_all_buildings() function

    prj.modelica_info.current_solver = "cvode"
    prj.modelica_info.interval_output = 900
    prj.modelica_info.start_time = 30326400
    prj.modelica_info.stop_time = 63072000

    prj.export_aixlib(internal_id=None, path=None)
    workspace = os.path.join("D:\\", "workspace")
    sim.queue_simulation(
        sim_function=sim.simulate,
        prj=prj,
        results_path=workspace,
        number_of_workers=1,
        start_time=prj.modelica_info.start_time,
        stop_time=prj.modelica_info.stop_time,
        output_interval=prj.modelica_info.interval_output,
        method=prj.modelica_info.current_solver,
        tolerance=0.0001,
    )

    print("Example 1: That's it! :)")

    # res_path = "D:\\dymola\\SimpleDistrictBuilding.mat"
    #
    # results_to_csv(res_path)

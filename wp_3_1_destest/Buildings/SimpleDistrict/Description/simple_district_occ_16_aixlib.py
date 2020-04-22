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
import datetime
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
            name="SimpleDistrictBuilding_occ_{}".format(bldg_number),
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
            ["ExpandedPolystrenemOrEPS", 0.01, 26, 0.036, 1.47, 0.8, 0.9],
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
            ["ExpandedPolystrenemOrEPS", 0.03, 26, 0.036, 1.47, 0.8, 0.9],
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
            ["Glasswool", 0.04, 80, 0.04, 0.84, 0.85, 0.9],
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

            # for an implementation, window area is missing

            # for win in zone.windows:
            #     win.area = 22.4
            #     win.layer = None
            #     for lay in layers_ow:
            #         temp_layer = Layer(parent=win)
            #         temp_layer.thickness = lay[1]
            #         temp_layer_material = Material(parent=temp_layer)
            #         temp_layer_material.name = lay[0]
            #         temp_layer_material.density = lay[2]
            #         temp_layer_material.thermal_conduc = lay[3]
            #         temp_layer_material.heat_capac = lay[4]
            #         temp_layer_material.solar_absorp = lay[5]
            #         temp_layer_material.ir_emissivity = lay[6]

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

    # To export the ready-to-run models simply call Project.export_aixlib().
    # You can specify the path, where the model files should be saved.
    # None means, that the default path in your home directory
    # will be used. If you only want to export one specific building, you can
    # pass over the internal_id of that building and only this model will be
    # exported. In this case we want to export all buildings to our home
    # directory, thus we are passing over None for both parameters.

    # final parameter Real[3] QDay(unit="W/m2") = {8,20,2}
    #     "Specific power for dayzone {day, evening, night}";
    #   final parameter Real[3] QNight(unit="W/m2") = {1.286,1.857,6}
    #     "Specific power for nightzone {day, evening, night}";
    #   final parameter Real[3] TDay(unit="degC") = {16,21,18}
    #     "Temperature set-points for dayzone {day, evening, night}";
    #   final parameter Real[3] TNight(unit="degC") = {16,18,20}
    #     "Temperature set-points for nightzone {day, evening, night}";
    #         With the first value daily between 7am and 5 pm, the second value
    #         between 5 pm and 11 pm and the third value during night.

    occ_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Occupants",
        "Profiles",
    )

    for i, bldg in enumerate(prj.buildings):
        # Load set temperature add 273.15 to be in K
        if i < 9:
            i = "0{}".format(i + 1)
        else:
            i = str(i + 1)
        set_temp_living = pd.read_csv(
            filepath_or_buffer=os.path.join(occ_path, "0{}_sh_day.txt".format(i)),
            index_col=0,
            skiprows=2,
            sep=" ",
            names=["values"],
        )
        set_temp_living.index = pd.date_range(
            datetime.datetime(2018, 1, 1),
            periods=len(set_temp_living["values"]),
            freq="15min",
        )
        set_temp_living = set_temp_living.resample("H").mean()
        set_temp_living.loc[:, "values"] = set_temp_living["values"] + 273.15
        set_temp_living = set_temp_living.round(2)

        set_temp_bed_room = pd.read_csv(
            filepath_or_buffer=os.path.join(occ_path, "0{}_sh_night.txt".format(i)),
            index_col=0,
            skiprows=2,
            sep=" ",
            names=["values"],
        )
        set_temp_bed_room.index = pd.date_range(
            datetime.datetime(2018, 1, 1),
            periods=len(set_temp_bed_room["values"]),
            freq="15min",
        )
        set_temp_bed_room = set_temp_bed_room.resample("H").mean()
        set_temp_bed_room.loc[:, "values"] = set_temp_bed_room["values"] + 273.15
        set_temp_bed_room = set_temp_bed_room.round(2)
        # Load convective and radiative internal loads.
        # Assumptions:

        # Convetive internal load is only modelled as Persons, the relative profile is
        # calcualted with profile/maximum

        # Radiative internal load is only modelled as machines, the relative profile is
        # calculated with profile/maximum

        # The number of persons is calculated using the maximum value, divided by 100
        # W/Person divided by total area resulting in Person/m², this is subdivided
        # 50/50 to day and night zone with same relative profile

        # The specific power of machines is calculated using the maximum value, divided
        # by total area resulting in Person/m², this is subdivided 50/50
        # to day and night zone with same relative profile

        q_con = pd.read_csv(
            filepath_or_buffer=os.path.join(occ_path, "0{}_QCon.txt".format(i)),
            index_col=0,
            skiprows=2,
            sep=" ",
            names=["values"],
        )
        q_con.index = pd.date_range(
            datetime.datetime(2018, 1, 1), periods=len(q_con["values"]), freq="15min"
        )
        q_con = q_con.resample("H").mean()
        q_con.loc[:, "rel_profile"] = q_con["values"] / q_con["values"].max()
        pers = q_con["values"].max() / 100 / bldg.net_leased_area  # Pers/m²
        q_con = q_con.round(2)
        q_rad = pd.read_csv(
            filepath_or_buffer=os.path.join(occ_path, "0{}_QRad.txt".format(i)),
            index_col=0,
            skiprows=2,
            sep=" ",
            names=["values"],
        )
        q_rad.index = pd.date_range(
            datetime.datetime(2018, 1, 1), periods=len(q_rad["values"]), freq="15min"
        )
        q_rad = q_rad.resample("H").mean()
        q_rad.loc[:, "rel_profile"] = q_rad["values"] / q_rad["values"].max()
        machines = q_rad["values"].max() / bldg.net_leased_area
        q_rad = q_rad.round(2)
        bldg.thermal_zones[0].use_conditions.heating_profile = set_temp_living[
            "values"
        ].values.tolist()

        bldg.thermal_zones[1].use_conditions.heating_profile = set_temp_bed_room[
            "values"
        ].values.tolist()

        bldg.thermal_zones[0].use_conditions.persons = pers * 0.5
        bldg.thermal_zones[0].use_conditions.fixed_heat_flow_rate_persons = 100
        bldg.thermal_zones[0].use_conditions.ratio_conv_rad_persons = 0.999
        bldg.thermal_zones[0].use_conditions.persons_profile = q_con[
            "rel_profile"
        ].values.tolist()

        bldg.thermal_zones[0].use_conditions.machines = machines * 0.5  # W/m²
        bldg.thermal_zones[0].use_conditions.ratio_conv_rad_machines = 0.001
        bldg.thermal_zones[0].use_conditions.machines_profile = q_rad[
            "rel_profile"
        ].values.tolist()

        bldg.thermal_zones[0].use_conditions.lighting_power = 0
        bldg.thermal_zones[0].infiltration_rate = 0.4
        bldg.thermal_zones[0].use_conditions.use_constant_ach_rate = True
        bldg.thermal_zones[0].use_conditions.base_ach = 0.4

        bldg.thermal_zones[1].use_conditions.persons = pers * 0.5
        bldg.thermal_zones[1].use_conditions.fixed_heat_flow_rate_persons = 100
        bldg.thermal_zones[1].use_conditions.ratio_conv_rad_persons = 0.999
        bldg.thermal_zones[1].use_conditions.persons_profile = q_con[
            "rel_profile"
        ].values.tolist()

        bldg.thermal_zones[1].use_conditions.machines = machines * 0.5  # W/m²
        bldg.thermal_zones[1].use_conditions.ratio_conv_rad_machines = 0.001
        bldg.thermal_zones[1].use_conditions.machines_profile = q_rad[
            "rel_profile"
        ].values.tolist()

        bldg.thermal_zones[1].use_conditions.lighting_power = 0
        bldg.thermal_zones[1].infiltration_rate = 0.4
        bldg.thermal_zones[1].use_conditions.use_constant_ach_rate = True
        bldg.thermal_zones[1].use_conditions.base_ach = 0.4

        # profiles for day and night zone representing the share of total number
        # of persons
        bldg.thermal_zones[0].model_attr.heat_load = 8024.46
        bldg.thermal_zones[1].model_attr.heat_load = 8548.50

    return prj


def results_to_csv(res_path):
    """
    This function loads the mat file and save it to csv.

    It loads the dymola result mat file and saves the indoor air temp of
    the two modelled zones and the total heating power in W.
    """
    res_all = pd.DataFrame()

    signals = [
        "Time",
        "multizone.PHeater[1]",
        "multizone.PHeater[2]",
        "multizone.TAir[1]",
        "multizone.TAir[2]",
    ]

    dymola = DymolaInterface()
    print("Reading signals: ", signals)

    dym_res = dymola.readTrajectory(
        fileName=res_path,
        signals=signals,
        rows=dymola.readTrajectorySize(fileName=res_path),
    )
    results = pd.DataFrame().from_records(dym_res).T
    results = results.rename(columns=dict(zip(results.columns.values, signals)))
    results.index = results["Time"]

    results["AixLib_Heating_Power_W"] = (
        results["multizone.PHeater[1]"] + results["multizone.PHeater[2]"]
    )

    # drop Time and single zones columns
    results = results.drop(["Time"], axis=1)

    results = results.rename(
        index=str,
        columns={
            "multizone.TAir[1]": "AixLib_T_dayzone",
            "multizone.TAir[2]": "AixLib_T_nightzone",
        },
    )

    # results = results.drop(index_to_drop)
    # results = results.groupby(level=0).first()
    # results.to_csv(path=res_path, delimiter=';')
    dymola.close()

    time = pd.to_numeric(results.index)
    time -= 31536000
    results.index = time
    results = results.ix[0:31536000]

    res_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "AixLib_SingleBuilding.csv"
    )

    results.to_csv(res_csv)

    print(results)
    print(res_csv)

    return results


if __name__ == "__main__":

    # test for new dataclasses for inas materials and typebuildings
    # not working. can investigate further sadly OS seems to be useless

    # from teaser.data.dataclass import DataClass

    # belg_type_elements = DataClass()
    # belg_type_elements.element_bind = None
    # belg_type_elements.path_tb = os.path.join(
    #     os.path.dirname(
    #         os.path.abspath(__file__)),
    #     "Specifications",
    #     "Belgium_TypeBuildingElements.xml")

    # belg_type_elements.path_mat = os.path.join(
    #     os.path.dirname(
    #         os.path.abspath(__file__)),
    #     "Specifications",
    #     "Belgium_MaterialTemplates.xml")

    # belg_type_elements.load_mat_binding()
    # belg_type_elements.load_tb_binding()
    prj = Project(load_data=True)
    prj.name = "Simple_District_Occ_Destest_AixLib"
    prj.used_library_calc = "AixLib"
    prj.number_of_elements_calc = 2
    # prj.weather_file_path = os.path.join(
    #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #     "Climate",
    #     "BEL_Brussels.064510_IWEC.mos",
    # )

    prj = example_generate_simple_district_building(prj=prj, nr_of_bldg=16)

    # To make sure the parameters are calculated correctly we recommend to
    # run calc_all_buildings() function

    prj.modelica_info.current_solver = "cvode"
    prj.modelica_info.interval_output = 900
    prj.modelica_info.start_time = 30326400
    prj.modelica_info.stop_time = 63072000

    prj.export_aixlib(internal_id=None, path=None)
    workspace = os.path.join("D:\\", "workspace")
    sim.queue_simulation(sim_function=sim.simulate, prj=prj, results_path=workspace)

    print("Example 1: That's it! :)")

    # res_path = "D:\\dymola\\SimpleDistrictBuilding.mat"
    #
    # results_to_csv(res_path)

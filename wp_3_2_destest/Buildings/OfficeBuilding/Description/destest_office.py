# -*- coding: utf-8 -*-
# @Author: MichaMans
# @Date:   2018-09-20 11:05:56
# @Last Modified by:   MichaMans
# @Last Modified time: 2020-01-20 15:00:25

"""Modelgeneration for Passauer Strasse."""

import os
from collections import OrderedDict
from teaser.project import Project
from teaser.logic.buildingobjects.building import Building
from teaser.logic.buildingobjects.thermalzone import ThermalZone
from teaser.logic.buildingobjects.useconditions import (
    UseConditions,
)
from teaser.logic.buildingobjects.buildingphysics.outerwall import OuterWall
from teaser.logic.buildingobjects.buildingphysics.floor import Floor
from teaser.logic.buildingobjects.buildingphysics.rooftop import Rooftop
from teaser.logic.buildingobjects.buildingphysics.groundfloor import GroundFloor
from teaser.logic.buildingobjects.buildingphysics.ceiling import Ceiling
from teaser.logic.buildingobjects.buildingphysics.window import Window
from teaser.logic.buildingobjects.buildingphysics.innerwall import InnerWall
from teaser.logic.buildingobjects.buildingphysics.layer import Layer
from teaser.logic.buildingobjects.buildingphysics.material import Material
import teaser.logic.utilities as utils

import pandas as pd
import datetime
import numpy as np


dir_this = os.path.abspath(os.path.dirname(__file__))
dir_src = os.path.abspath(os.path.dirname(dir_this))

"""
ATTENTION:

This script need branch issue438_cooler of TEASER

THis script creates the DESTEST Office Building

"""

BUILDING_1 = {
    "year_of_construction": 2010,
    "number_of_floors": 1,
    "height_of_floors": 3.0,
    "net_leased_area": 1512,
    # opening hours: {day_of_week:[open, close]} close = real closing time - 1h
    "opening_hours_retail": None,
    # todo fill with values
    "zone_area_factors": {
        "Office_North": {
            "existing": True,
            "is_groundfloor": True,
            "is_upperfloor": True,
            "exist_n_times": 1,
            "usage": "Group Office (between 2 and 6 employees)",
            "area": 550,
            "windows": [
                [194.4, 90, 0],
                [28.35, 90, 90],
                [28.35, 90, 270],
            ],
            "outer_walls": [
                [21.6, 90, 0],
                [3.15, 90, 90],
                [3.15, 90, 270],
            ],
            "inner_wall": 810,
            "floor": [550, 0, -2],
            "ceiling": [550, 0, -1],
            "ahu": False,
            "T_set_heat": 20,
            "T_set_cool": 22,
            "night_set_back": 0,
            "shading_g_total": 0.3,
            "shading_max_irr": 100,
        },
        "Office_South": {
            "existing": True,
            "is_groundfloor": True,
            "is_upperfloor": True,
            "exist_n_times": 1,
            "usage": "Group Office (between 2 and 6 employees)",
            "area": 550,
            "windows": [
                [28.35, 90, 90],
                [194.4, 90, 180],
                [28.35, 90, 270],
            ],
            "outer_walls": [
                [3.15, 90, 90],
                [28.35, 90, 180],
                [3.15, 90, 270],
            ],
            "inner_wall": 810,
            "floor": [550, 0, -2],
            "ceiling": [550, 0, -1],
            "ahu": False,
            "T_set_heat": 20,
            "T_set_cool": 22,
            "night_set_back": 0,
            "shading_g_total": 0.3,
            "shading_max_irr": 100,
        },
        "Traffic area": {
            "existing": True,
            "isRetail": False,
            "is_groundfloor": True,
            "is_upperfloor": True,
            "exist_n_times": 1,
            "usage": "Traffic area",
            "area": 411,
            # cannot be null
            "windows": [[0.0001, 90, 225]],
            "outer_walls": [[0.0001, 90, 225]],
            "inner_wall": 605,
            "floor": [411, 0, -2],
            "ceiling": [411, 0, -1],
            "ahu": False,
            "T_set_heat": 15,
            "T_set_cool": 22,
            "night_set_back": 0,
            "shading_g_total": 0.3,
            "shading_max_irr": 100,
        },
    },
    "element_construction": {
        "office": {
            "outer_wall": {
                "layer_thickness": [0.250, 0.160, 0.010],
                "materials": [
                    "concrete_CEM_I_425R_wz05",
                    "EPS_perimeter_insulation_top_layer",
                    "lime_cement_plaster",
                ],
            },
            "windows": {"type": "Waermeschutzverglasung, dreifach", "g_value": 0.4},
            # todo fill up values for rooftop
            "rooftop": {
                "layer_thickness": [0.2, 0.2],
                "materials": ["concrete_CEM_I_425R_wz05", "Multipor_insulation"],
            },
            "groundfloor": {
                "layer_thickness": [0.400, 0.1],
                "materials": [
                    "concrete_CEM_I_425R_wz05",
                    "EPS_perimeter_insulation_top_layer",
                ],
            },
        },
        # todo: fill with values for retail
        "retail": {
            "outer_wall": {
                "layer_thickness": [0.250, 0.160, 0.010],
                "materials": [
                    "concrete_CEM_I_425R_wz05",
                    "EPS_perimeter_insulation_top_layer",
                    "lime_cement_plaster",
                ],
            },
            "windows": {"type": "Waermeschutzverglasung, dreifach", "g_value": 0.4},
            # todo fill up values for rooftop
            "rooftop": {
                "layer_thickness": [0.2, 0.2],
                "materials": ["concrete_CEM_I_425R_wz05", "Multipor_insulation"],
            },
            "groundfloor": {
                "layer_thickness": [0.400, 0.1],
                "materials": [
                    "concrete_CEM_I_425R_wz05",
                    "EPS_perimeter_insulation_top_layer",
                ],
            },
        },
    },
}

# We need to account for building rotation

# We need to account for Window share of outer wall


def generate_dict_as_building(prj, name, dict):
    """
    The function creates a floor as a building.

    With the given thermal zones as an input
    """
    # todo: integrate function to sum up zones n times and to check for
    # todo ground and upper floor to add roof and floor plate. here we have to
    # todo make sure that we have only one roof and floor plate per building!
    prj.modelica_info.weekday = 3  # 0-Monday, 6-Sunday
    prj.modelica_info.simulation_start = 0  # start time for simulation
    bldg = Building(parent=prj)
    bldg.name = name
    bldg.opening_hours_retail = dict["opening_hours_retail"]
    bldg.year_of_construction = dict["year_of_construction"]
    bldg.number_of_floors = dict["number_of_floors"]
    bldg.height_of_floors = dict["height_of_floors"]
    bldg.net_leased_area = dict["net_leased_area"]
    bldg.with_ahu = False

    for key in dict["zone_area_factors"].keys():
        if dict["zone_area_factors"][key]["existing"]:
            for x in range(0, dict["zone_area_factors"][key]["exist_n_times"]):
                tz = ThermalZone(parent=bldg)
                tz.name = key + "_floor_" + str(x)
                tz.area = dict["zone_area_factors"][key]["area"]
                tz.volume = tz.area * bldg.number_of_floors * bldg.height_of_floors
                tz.infiltration_rate = 0.5
                tz.use_conditions = UseConditions(parent=tz)
                tz.use_conditions.load_use_conditions(
                    dict["zone_area_factors"][key]["usage"], prj.data
                )
                tz.use_conditions.with_ahu = dict["zone_area_factors"][key]["ahu"]
                tz.use_conditions.set_temp_cool = (
                    dict["zone_area_factors"][key]["T_set_cool"] + 273.15
                )
                tz.use_conditions.set_temp_heat = (
                    dict["zone_area_factors"][key]["T_set_heat"] + 273.15
                )
                tz.use_conditions.shading_max_irr = dict["zone_area_factors"][key][
                    "shading_max_irr"
                ]
                tz.use_conditions.shading_g_total = dict["zone_area_factors"][key][
                    "shading_g_total"
                ]

                if key == "Residential":
                    tz.use_conditions.persons = 0.0338

                construction_dict = dict["element_construction"]["office"]
                for wall in dict["zone_area_factors"][key]["outer_walls"]:
                    out_wall = OuterWall(parent=tz)
                    out_wall.name = "outer_wall_" + str(wall[2])
                    out_wall.area = wall[0]
                    out_wall.tilt = wall[1]
                    out_wall.orientation = wall[2]

                    for i, layer_thickness in enumerate(
                        construction_dict["outer_wall"]["layer_thickness"]
                    ):
                        layer = Layer(parent=out_wall)
                        layer.thickness = layer_thickness
                        material = Material(parent=layer)
                        material.load_material_template(
                            mat_name=construction_dict["outer_wall"]["materials"][i],
                            data_class=prj.data,
                        )

                in_wall = InnerWall(parent=tz)
                in_wall.name = "inner_wall"
                in_wall.load_type_element(
                    year=bldg.year_of_construction, construction="light"
                )
                in_wall.area = dict["zone_area_factors"][key]["inner_wall"]
                # add groundfloor or normal floor depending on given parameters
                if x == 0 and dict["zone_area_factors"][key]["is_groundfloor"]:
                    groundfloor = GroundFloor(parent=tz)
                    groundfloor.name = "groundfloor_" + str(
                        dict["zone_area_factors"][key]["floor"][2]
                    )
                    groundfloor.area = dict["zone_area_factors"][key]["floor"][0]
                    groundfloor.tilt = dict["zone_area_factors"][key]["floor"][1]
                    groundfloor.orientation = dict["zone_area_factors"][key]["floor"][2]
                    for i, layer_thickness in enumerate(
                        construction_dict["groundfloor"]["layer_thickness"]
                    ):
                        layer = Layer(parent=groundfloor)
                        layer.thickness = layer_thickness
                        material = Material(parent=layer)
                        material.load_material_template(
                            mat_name=construction_dict["groundfloor"]["materials"][i],
                            data_class=prj.data,
                        )
                else:
                    floor = Floor(parent=tz)
                    floor.name = "floor_" + str(
                        dict["zone_area_factors"][key]["floor"][2]
                    )
                    floor.load_type_element(
                        year=bldg.year_of_construction, construction="heavy"
                    )
                    floor.area = dict["zone_area_factors"][key]["floor"][0]
                    floor.tilt = dict["zone_area_factors"][key]["floor"][1]
                    floor.orientation = dict["zone_area_factors"][key]["floor"][2]
                if x == 0 and dict["zone_area_factors"][key]["is_upperfloor"]:
                    rooftop = Rooftop(parent=tz)
                    rooftop.name = "rooftop" + str(
                        dict["zone_area_factors"][key]["ceiling"][2]
                    )
                    rooftop.area = dict["zone_area_factors"][key]["ceiling"][0]
                    rooftop.tilt = dict["zone_area_factors"][key]["ceiling"][1]
                    rooftop.orientation = dict["zone_area_factors"][key]["ceiling"][2]
                    for i, layer_thickness in enumerate(
                        construction_dict["rooftop"]["layer_thickness"]
                    ):
                        layer = Layer(parent=rooftop)
                        layer.thickness = layer_thickness
                        material = Material(parent=layer)
                        material.load_material_template(
                            mat_name=construction_dict["rooftop"]["materials"][i],
                            data_class=prj.data,
                        )
                else:
                    ceiling = Ceiling(parent=tz)
                    ceiling.name = "ceiling_" + str(
                        dict["zone_area_factors"][key]["ceiling"][2]
                    )
                    ceiling.load_type_element(
                        year=bldg.year_of_construction, construction="heavy"
                    )
                    ceiling.area = dict["zone_area_factors"][key]["ceiling"][0]
                    ceiling.tilt = dict["zone_area_factors"][key]["ceiling"][1]
                    ceiling.orientation = dict["zone_area_factors"][key]["ceiling"][2]

                for win in dict["zone_area_factors"][key]["windows"]:
                    window = Window(parent=tz)
                    window.name = "outer_wall_" + str(win[2])
                    window.load_type_element(
                        year=bldg.year_of_construction, construction="EnEv"
                    )
                    window.area = win[0]
                    window.tilt = win[1]
                    window.orientation = win[2]
                    window.g_value = construction_dict["windows"]["g_value"]

    if bldg.with_ahu is True:

        profile_temperature_ahu_summer = 7 * [293.15] + 12 * [287.15] + 5 * [293.15]
        profile_temperature_ahu_winter = 24 * [299.15]
        bldg.central_ahu.profile_temperature = (
            120 * profile_temperature_ahu_winter
            + 124 * profile_temperature_ahu_summer
            + 121 * profile_temperature_ahu_winter
        )
        bldg.central_ahu.profile_min_relative_humidity = 8760 * [0.45]
        bldg.central_ahu.profile_max_relative_humidity = 8760 * [0.65]
        bldg.central_ahu.profile_v_flow = (7 * [0.0] + 12 * [1.0] + 5 * [0.0]) * 365
        # bldg.central_ahu.profile_temperature = (
        #     7 * [293.15] + 12 * [295.15] + 6 * [293.15])
        # # according to :cite:`DeutschesInstitutfurNormung.2016`
        # bldg.central_ahu.profile_min_relative_humidity = (25 * [0.45])
        # #  according to :cite:`DeutschesInstitutfurNormung.2016b`  and
        # # :cite:`DeutschesInstitutfurNormung.2016`
        # bldg.central_ahu.profile_max_relative_humidity = (25 * [0.65])
        # bldg.central_ahu.profile_v_flow = (
        #     7 * [0.0] + 12 * [1.0] + 6 * [0.0])
        # according to user
        # profile in :cite:`DeutschesInstitutfurNormung.2016`
        bldg.central_ahu.heat_recovery = True
        bldg.central_ahu.efficiency_recovery = 0.8

    bldg.calc_building_parameter()

    # account for cooling load
    for tz in bldg.thermal_zones:
        spec_cool_load = -100 * tz.area
        tz.model_attr.cool_load = tz.area * spec_cool_load * 1.3
        tz.model_attr.heat_load = tz.model_attr.heat_load * 1.9
    # bldg.library_attr.use_set_back_cool = True

    return prj


if __name__ == "__main__":
    prj = Project(load_data=True)
    prj.name = "DESTEST Office"
    prj.data.load_uc_binding()
    prj.weather_file_path = os.path.join(
        dir_src,
        "Climate",
        "BEL_Brussels.064510_IWEC.mos",
    )
    prj = generate_dict_as_building(prj, "Building_1", BUILDING_1)

    prj.modelica_info.current_solver = "dassl"
    # prj.calc_all_buildings()

    prj.save_project("DESTEST", os.path.join(dir_src, "Description"))

    path = prj.export_aixlib()

    print("DESTEST OFFICE: That's it! :)")

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xlrd
import math
import statistics
import random
import matplotlib.dates as mdates

import pycity_base.classes.demand.domestic_hot_water as dhw
import pycity_base.classes.timer as time
import pycity_base.classes.weather as weath
import pycity_base.classes.prices as price
import pycity_base.classes.environment as env
import pycity_base.classes.demand.occupancy as occ


def main():

    plot_average_profiles_pycity()

    x, water_pycity_alias_60 = generate_dhw_profile_pycity_alias(s_step=60)
    x, water_pycity_alias_600 = generate_dhw_profile_pycity_alias(s_step=600)

    x, water_pycity_60 = generate_dhw_profile_pycity(s_step=60)
    x, water_pycity_600 = generate_dhw_profile_pycity(s_step=600)

    x, water_dhwcalc_60 = import_from_dhwcalc(s_step=60)
    x, water_dhwcalc_600 = import_from_dhwcalc(s_step=600)

    pass


def import_from_dhwcalc(s_step=60, temp_dT=35, print_stats=True,
                        plot_demand=True, plot_sliced_demand=True,
                        start_plot='2019-08-01', end_plot='2019-08-14',
                        save_fig=True):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour). To get
    Energyflows -> Q = Vdot * rho * cp * dT

    :param  s_step:     int:    resolution of output file in seconds
    :param  temp_dT:    int:    average Temperature Difference between
                                Freshwater and DHW
    :return dhw_demand: list:   each timestep contains the Energyflow in [W]
    """

    # timeseries are 200 L/d -> 73000 L/a (for 5 people, each 40 L/d)
    if s_step == 60:
        dhw_file = "DHWcalc_200L_1min_1cat_step_functions_DHW.txt"
    elif s_step == 600:
        dhw_file = "DHWcalc_200L_10min_1cat_step_functions_DHW.txt"
    else:
        raise Exception("Unkown Time Step for DHWcalc")
    dhw_profile = Path.cwd() / dhw_file

    # Flowrates
    water_LperH = [int(word.strip('\n')) for word in
                   open(dhw_profile).readlines()]  # L/h each step
    water_LperSec = [x / 3600 for x in water_LperH]  # L/s each step

    rho = 980 / 1000  # kg/L for Water (at 60°C? at 10°C its = 1)
    cp = 4180  # J/kgK
    dhw_demand = [i * rho * cp * temp_dT for i in water_LperSec]  # in W

    # compute Sums and Maxima for Water and Heat
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
    max_water_flow = round(max(water_LperH), 1)  # in L/h
    yearly_dhw_demand = round(sum(dhw_demand) * s_step / (3600 * 1000), 1) # kWh
    max_dhw_heat_flow = round(max(dhw_demand) / 1000, 1)  # in kW

    if print_stats:

        print("Yearly drinking water demand from DHWcalc is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from DHWcalc is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    if plot_demand:

        fig, ax = plt.subplots()
        # ax.plot(dhw_demand, linewidth=0.7, label="Heat")
        ax.plot(water_LperH, linewidth=0.7, label="Water")
        plt.ylabel('Water [L/h]')
        plt.xlabel('Timesteps in a year, length = {}s'.format(s_step))
        plt.title('Water and Heat time-series from DHWcalc, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow, yearly_dhw_demand,
            max_dhw_heat_flow))

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_DHWcalc.pdf")

    if plot_sliced_demand:
        # not fully working yet, some problems with resample delta

        # RWTH colours
        rwth_blue = "#00549F"
        rwth_red = "#CC071E"

        # sns.set_style("white")
        sns.set_context("paper")

        timedelta = pd.Timedelta(pd.Timestamp(end_plot) - pd.Timestamp(
            start_plot))

        resample_delta = str(s_step) + 'S'

        # set date range to simplify plot slicing
        date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                                   freq=str(s_step) + 'S')
        date_range = date_range[:-1]

        # make dataframe for plotting with seaborn
        plot_df = pd.DataFrame({'Waterflow [L/h]': water_LperH},
                               index=date_range)

        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(data=plot_df[start_plot:end_plot],
                           linewidth=1.2, palette=[rwth_red])

        plt.legend()

        plt.title('Water and Heat time-series from DHWcalc, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow,  yearly_dhw_demand,
            max_dhw_heat_flow))

        # set the x axis ticks
        # https://matplotlib.org/3.1.1/gallery/
        # ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[3] = '%d-%b'
        formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y',
                                    '%d %b %Y %H:%M', ]
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_DHWcalc_sliced.pdf")

    return dhw_demand, water_LperH


def generate_dhw_profile_dhwcalc_alias(initial_day=0, s_step=60, temp_dT=35,
                                       print_stats=True, plot_demand=True,
                                       save_fig=False, ):

    # generate probability for weekday with 6 step functions

    step_0_wd = 6.5
    p_0_wd = 0.01

    step_1_wd = 1
    p_1_wd = 0.5

    step_2_wd = 4.5
    p_2_wd = 0.06

    step_3_wd = 1
    p_3_wd = 0.16

    step_4_wd = 5
    p_4_wd = 0.06

    step_5_wd = 4
    p_5_wd = 0.2

    step_6_wd = 2
    p_6_wd = 0.01

    steps_wd = [step_0_wd, step_1_wd, step_2_wd, step_3_wd, step_4_wd,
                step_5_wd, step_6_wd]
    ps_wd = [p_0_wd, p_1_wd, p_2_wd, p_3_wd, p_4_wd, p_5_wd, p_6_wd]

    assert sum(steps_wd) == 24
    assert sum(ps_wd) == 1

    p_0_wd_lst = [p_0_wd for i in range(int(step_0_wd * 60))]
    p_1_wd_lst = [p_1_wd for i in range(int(step_1_wd * 60))]
    p_2_wd_lst = [p_2_wd for i in range(int(step_2_wd * 60))]
    p_3_wd_lst = [p_3_wd for i in range(int(step_3_wd * 60))]
    p_4_wd_lst = [p_4_wd for i in range(int(step_4_wd * 60))]
    p_5_wd_lst = [p_5_wd for i in range(int(step_5_wd * 60))]
    p_6_wd_lst = [p_6_wd for i in range(int(step_6_wd * 60))]

    ps_lsts = [p_0_wd_lst, p_1_wd_lst, p_2_wd_lst, p_3_wd_lst, p_4_wd_lst,
               p_5_wd_lst, p_6_wd_lst]

    p_wd = []

    for lst in ps_lsts:
        p_wd.extend(lst)

    # generate probability for weekend with 6 step functions

    step_0_we = 7
    p_0_we = 0.02

    step_1_we = 2
    p_1_we = 0.475

    step_2_we = 6
    p_2_we = 0.071

    step_3_we = 2
    p_3_we = 0.237

    step_4_we = 3
    p_4_we = 0.036

    step_5_we = 3
    p_5_we = 0.143

    step_6_we = 1
    p_6_we = 0.018

    steps_we = [step_0_we, step_1_we, step_2_we, step_3_we, step_4_we,
                step_5_we, step_6_we]
    ps_we = [p_0_we, p_1_we, p_2_we, p_3_we, p_4_we, p_5_we, p_6_we]

    assert sum(steps_we) == 24
    assert sum(ps_we) == 1

    p_0_we_lst = [p_0_we for i in range(int(step_0_we * 60))]
    p_1_we_lst = [p_1_we for i in range(int(step_1_we * 60))]
    p_2_we_lst = [p_2_we for i in range(int(step_2_we * 60))]
    p_3_we_lst = [p_3_we for i in range(int(step_3_we * 60))]
    p_4_we_lst = [p_4_we for i in range(int(step_4_we * 60))]
    p_5_we_lst = [p_5_we for i in range(int(step_5_we * 60))]
    p_6_we_lst = [p_6_we for i in range(int(step_6_we * 60))]

    ps_lsts = [p_0_we_lst, p_1_we_lst, p_2_we_lst, p_3_we_lst, p_4_we_lst,
               p_5_we_lst, p_6_we_lst]

    p_we = []

    for lst in ps_lsts:
        p_we.extend(lst)

    # generate average profile
    l_day = 200

    average_profile = [random.gauss(l_day, sigma=114.33) for i in range(1440)]
    average_profile = [abs(entry) for entry in average_profile]

    # ab hier wie in PyCity Alias:

    # time series for return statement
    water = []  # in L/h
    heat = []  # in W

    number_days = 365

    for day in range(number_days):

        # Is the current day on a weekend?
        if (day + initial_day) % 7 >= 5:
            p_day = p_we
        else:
            p_day = p_wd

        water_daily = []

        # Compute seasonal factor
        arg = math.pi * (2 / 365 * day - 1 / 4)
        probability_season = 1 + 0.1 * np.cos(arg)

        timesteps_day = int(24 * 3600 / s_step)

        for t in range(timesteps_day):  # Iterate over all time-steps in a day

            # Compute probability for tap water demand at time t
            probability = p_day[t] * probability_season

            # Check if tap water demand occurs. The higher the probability,
            # the more likely the if statement is true.
            if random.random() < probability:
                # Compute amount of tap water consumption. Start with seed?
                # This consumption has to be positive!
                water_t = random.gauss(average_profile[t], sigma=114.33)
                water_daily.append(abs(water_t))
            else:
                water_daily.append(0)

        c = 4180  # J/(kg.K)
        rho = 0.980  # kg/l
        heat_daily = [i * rho * c * temp_dT / s_step for i in water_daily]  # W

        # Include current_water and current_heat in water and heat
        water.extend(water_daily)
        heat.extend(heat_daily)

    water_LperH = water
    water_LperSec = [x / 3600 for x in water_LperH]

    # compute Sums and Maxima for Water and Heat
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
    max_water_flow = round(max(water_LperH), 1)  # in L/h
    yearly_dhw_demand = round(sum(heat) * s_step / (3600 * 1000), 1)  # kWh
    max_dhw_heat_flow = round(max(heat) / 1000, 1)  # in kW
    av_water_flow = statistics.mean(water_LperH)

    av_water_flow_lst = [av_water_flow for i in range(365 * 24 * s_step)]

    if print_stats:

        print("Yearly drinking water demand from PyCity Alias is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from PyCity Alias is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    if plot_demand:
        fig, ax = plt.subplots()
        ax.plot(water_LperH, linewidth=0.7, label="Water")
        ax.plot(av_water_flow_lst, linewidth=1, label="Average")
        plt.ylabel('Water [L/h]')
        plt.xlabel('Timesteps in a year, length = {}s'.format(s_step))
        plt.title('Water and Heat time-series from PyCity Alias, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow, yearly_dhw_demand,
            max_dhw_heat_flow))

        plt.legend(loc="upper left")
        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_DHWcalc_Alias.pdf")

    plot_demand_sliced = True
    if plot_demand_sliced:
        fig, ax = plt.subplots()

        end = int(2 * 24 * 3600 / s_step)

        ax.plot(water_LperH[0:end], linewidth=0.7, label="Water")
        ax.plot(av_water_flow_lst[0:end], linewidth=0.7, label="Average")
        plt.ylabel('Water [L/h]')
        plt.xlabel('Timesteps in a year, length = {}s'.format(s_step))
        plt.title('Water and Heat time-series from PyCity Alias, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow, yearly_dhw_demand,
            max_dhw_heat_flow))

        plt.legend(loc="upper left")
        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_DHWcalc_Alias.pdf")


    return heat, water_LperH


def generate_dhw_profile_pycity_alias(s_step=60, initial_day=0,
                                      current_occupancy=5, temp_dT=35,
                                      print_stats=True, plot_demand=True,
                                      save_fig=False):
    """
    :param: s_step: int:        seconds within a time step. Should be
                                60s for pycity.
    :param: initial_day:        0: Mon, 1: Tue, 2: Wed, 3: Thur,
                                4: Fri, 5 : Sat, 6 : Sun
    :param: current_occuapncy:  number of people in the house. In PyCity,
                                this is a list and the occupancy changes during
                                the year. Between 0 and 5. Values have to be
                                integers. occupancy of 6 people seems to be
                                wrongly implemented in PyCity, as the sum of
                                the probabilities increases with occupancy (
                                1-5) but then decreases for the 6th person.
    :param: temp_dT: int/float: How much does the tap water has to be heated up?
    :return: water: List:       Tap water volume flow in liters per hour.
             heat : List:       Resulting minute-wise sampled heat demand in
                                Watt. The heat capacity of water is assumed
                                to be 4180 J/(kg.K) and the density is
                                assumed to be 980 kg/m3.
    """

    # get dhw stochastical file should be in the same dir as this script.

    profiles_path = Path.cwd() / 'dhw_stochastical.xlsx'
    profiles = {"we": {}, "wd": {}}
    book = xlrd.open_workbook(profiles_path)

    # Iterate over all sheets. wd = weekday, we = weekend. mw = ist the
    # average profile. occupancy is between 1-6 (we1 - we6).
    for sheetname in book.sheet_names():
        sheet = book.sheet_by_name(sheetname)

        # Read values
        values = [sheet.cell_value(i, 0) for i in range(1440)]

        # Store values in dictionary
        if sheetname in ("wd_mw", "we_mw"):
            profiles[sheetname] = values  # minute-wise average profile L/h
        elif sheetname[1] == "e":
            profiles["we"][int(sheetname[2])] = values  # probabilities 0 - 1
        else:
            profiles["wd"][int(sheetname[2])] = values  # probabilities 0 - 1

    # https://en.wikipedia.org/wiki/Geometric_distribution
    # occupancy is random, not a function of daytime! -> reasonable?
    timesteps_year = int(365 * 24 * 3600 / s_step)
    occupancy = np.random.geometric(p=0.8, size=timesteps_year) - 1  # [0, 2..]
    occupancy = np.minimum(5, occupancy)

    # time series for return statement
    water = []  # in L/h
    heat = []  # in W

    number_days = 365

    for day in range(number_days):

        # Is the current day on a weekend?
        if (day + initial_day) % 7 >= 5:
            probability_profiles = profiles["we"]
            average_profile = profiles["we_mw"]
        else:
            probability_profiles = profiles["wd"]
            average_profile = profiles["wd_mw"]

        water_daily = []

        # Compute seasonal factor
        arg = math.pi * (2 / 365 * day - 1 / 4)
        probability_season = 1 + 0.1 * np.cos(arg)

        timesteps_day = int(24 * 3600 / s_step)
        for t in range(timesteps_day):  # Iterate over all time-steps in a day

            first_timestep_day = day * timesteps_day
            last_timestep_day = (day + 1) * timesteps_day
            daily_occupancy = occupancy[first_timestep_day:last_timestep_day]
            current_occupancy = daily_occupancy[t]

            if current_occupancy > 0:
                probability_profile = probability_profiles[current_occupancy][t]
            else:
                probability_profile = 0

            # Compute probability for tap water demand at time t
            probability = probability_profile * probability_season

            # Check if tap water demand occurs. The higher the probability,
            # the more likely the if statement is true.
            if random.random() < probability:
                # Compute amount of tap water consumption. Start with seed?
                # This consumption has to be positive!
                water_t = random.gauss(average_profile[t], sigma=114.33)
                water_daily.append(abs(water_t))
            else:
                water_daily.append(0)

        c = 4180  # J/(kg.K)
        rho = 0.980  # kg/l
        heat_daily = [i * rho * c * temp_dT / s_step for i in water_daily]  # W

        # Include current_water and current_heat in water and heat
        water.extend(water_daily)
        heat.extend(heat_daily)

    water_LperH = water
    water_LperSec = [x / 3600 for x in water_LperH]

    # compute Sums and Maxima for Water and Heat
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
    max_water_flow = round(max(water_LperH), 1)  # in L/h
    yearly_dhw_demand = round(sum(heat) * s_step / (3600 * 1000), 1)  # kWh
    max_dhw_heat_flow = round(max(heat) / 1000, 1)  # in kW

    if print_stats:

        print("Yearly drinking water demand from PyCity Alias is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from PyCity Alias is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    if plot_demand:
        fig, ax = plt.subplots()
        # ax.plot(dhw_demand, linewidth=0.7, label="Heat")
        ax.plot(water_LperH, linewidth=0.7, label="Water")
        plt.ylabel('Water [L/h]')
        plt.xlabel('Timesteps in a year, length = {}s'.format(s_step))
        plt.title('Water and Heat time-series from PyCity Alias, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow, yearly_dhw_demand,
            max_dhw_heat_flow))

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_PyCity_Alias.pdf")

    return heat, water_LperH


def generate_dhw_profile_pycity(s_step=60, temp_dT=35, print_stats=True,
                                plot_demand=True, save_fig=False):
    """
    from https://github.com/RWTH-EBC/pyCity
    :return:
    """
    #  Generate environment with timer, weather, and prices objects
    timer = time.Timer(time_discretization=s_step,  # in seconds
                       timesteps_total=int(365 * 24 * 3600 / s_step)
                       )

    weather = weath.Weather(timer=timer)
    prices = price.Prices()
    environment = env.Environment(timer=timer, weather=weather, prices=prices)

    #  Generate occupancy object with stochastic user profile
    occupancy = occ.Occupancy(environment=environment, number_occupants=5)

    dhw_obj = dhw.DomesticHotWater(
        environment=environment,
        t_flow=10 + temp_dT,  # DHW output temperature in degree Celsius
        method=2,  # Stochastic dhw profile, Method 1 not working
        supply_temperature=10,  # DHW inlet flow temperature in degree C.
        occupancy=occupancy.occupancy)  # Occupancy profile (600 sec resolution)
    dhw_demand = dhw_obj.loadcurve  # ndarray with 8760 timesteps in Watt

    # constants of pyCity:
    cp = 4180
    rho = 980 / 1000
    temp_diff = 35

    water_LperSec = [i / (rho * cp * temp_diff) for i in dhw_demand]
    water_LperH = [x * 3600 for x in water_LperSec]

    # compute Sums and Maxima for Water and Heat
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
    max_water_flow = round(max(water_LperH), 1)  # in L/h
    yearly_dhw_demand = round(sum(dhw_demand) * s_step / (3600 * 1000), 1) # kWh
    max_dhw_heat_flow = round(max(dhw_demand) / 1000, 1)  # in kW

    if print_stats:

        print("Yearly drinking water demand from PyCity is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from PyCity is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    if plot_demand:

        fig, ax = plt.subplots()
        # ax.plot(dhw_demand, linewidth=0.7, label="Heat")
        ax.plot(water_LperH, linewidth=0.7, label="Water")
        plt.ylabel('Water [L/h]')
        plt.xlabel('Timesteps in a year, length = {}s'.format(s_step))
        plt.title('Water and Heat time-series from PyCity, dT = {} °C\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            temp_dT, yearly_water_demand, max_water_flow, yearly_dhw_demand,
            max_dhw_heat_flow))

        plt.show()

        if save_fig:
            dir_output = Path.cwd() / "plots"
            dir_output.mkdir(exist_ok=True)
            fig.savefig(dir_output / "Demand_PyCity.pdf")

    return dhw_demand, water_LperH


def plot_average_profiles_pycity(save_fig=False):

    profiles_path = Path.cwd() / 'dhw_stochastical.xlsx'
    profiles = {"we": {}, "wd": {}}
    book = xlrd.open_workbook(profiles_path)

    s_step = 600

    # Iterate over all sheets. wd = weekday, we = weekend. mw = ist the
    # average profile in [L/h] in 10min steps. occupancy is between 1-6 (we1 -
    # we6).
    for sheetname in book.sheet_names():
        sheet = book.sheet_by_name(sheetname)

        # Read values
        values = [sheet.cell_value(i, 0) for i in range(1440)]

        # Store values in dictionary
        if sheetname in ("wd_mw", "we_mw"):
            profiles[sheetname] = values  # minute-wise average profile L/h

    water_LperH_we = profiles["we_mw"]
    water_LperH_wd = profiles["wd_mw"]

    water_L_we = [i * s_step / 3600 for i in water_LperH_we]
    water_L_wd = [i * s_step / 3600 for i in water_LperH_wd]

    daily_water_we = round(sum(water_L_we), 1)
    daily_water_wd = round(sum(water_L_wd), 1)

    av_wd_lst = [statistics.mean(water_LperH_we) for i in range(1440)]
    av_we_lst = [statistics.mean(water_LperH_wd) for i in range(1440)]

    fig, ax = plt.subplots()
    ax.plot(water_LperH_we, linewidth=0.7, label="Weekend")
    ax.plot(water_LperH_wd, linewidth=0.7, label="Weekday")
    ax.plot(av_wd_lst, linewidth=0.7, label="Average Weekday")
    ax.plot(av_we_lst, linewidth=0.7, label="Average Weekday")
    plt.ylabel('Water [L/h]')
    plt.xlabel('Minutes in a day')
    plt.title('Average profiles from PyCity. \n'
              'Daily Sum Weekday: {} L, Daily Sum Weekend: {} L'.format(
        daily_water_wd, daily_water_we))

    plt.legend(loc='upper left')
    plt.show()

    if save_fig:
        dir_output = Path.cwd() / "plots"
        dir_output.mkdir(exist_ok=True)
        fig.savefig(dir_output / "Average_Profiles_PyCity.pdf")


if __name__ == '__main__':
    main()

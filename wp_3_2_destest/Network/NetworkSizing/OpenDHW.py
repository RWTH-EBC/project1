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

    generate_dhw_profile_dhwcalc_alias(s_step=60)
    generate_dhw_profile_dhwcalc_alias(s_step=600)

    x, water_pycity_alias_60 = generate_dhw_profile_pycity_alias(s_step=60)
    x, water_pycity_alias_600 = generate_dhw_profile_pycity_alias(s_step=600)

    x, water_pycity_60 = generate_dhw_profile_pycity(s_step=60)
    x, water_pycity_600 = generate_dhw_profile_pycity(s_step=600)

    x, water_dhwcalc_60 = import_from_dhwcalc(s_step=60)
    x, water_dhwcalc_600 = import_from_dhwcalc(s_step=600)

    pass


def import_from_dhwcalc(s_step=60, temp_dT=35, print_stats=True,
                        plot_demand=True, start_plot='2019-08-01',
                        end_plot='2019-08-03', save_fig=True):
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

    # Flowrate in Liter per Hour in each Step
    water_LperH = [int(word.strip('\n')) for word in
                   open(dhw_profile).readlines()]  # L/h each step

    # Plot
    dhw_demand = compute_stats_and_plot_demand(
        method='DHWcalc',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )

    return dhw_demand, water_LperH


def generate_dhw_profile_dhwcalc_alias(initial_day=0, s_step=60, temp_dT=35,
                                       print_stats=True, plot_demand=True,
                                       start_plot='2019-08-01',
                                       end_plot='2019-08-03', save_fig=False):

    l_day = 200

    average_profile = [random.gauss(l_day, sigma=114.33) for i in range(1440)]
    average_profile = [abs(entry) for entry in average_profile]

    p_we = generate_daily_probability_step_function(mode='weekend',
                                                    s_step=s_step)
    p_wd = generate_daily_probability_step_function(mode='weekday',
                                                    s_step=s_step)

    # Probability Shift between weeday and weekend
    p_wd_to_we = 1.2

    p_wd_factor = 1 / (5 / 7 + p_wd_to_we * 2 / 7)
    p_we_factor = 1 / (1 / p_wd_to_we * 5 / 7 + 2 / 7)

    assert p_we_factor / p_wd_factor == p_wd_to_we

    p_we = [p * p_we_factor for p in p_we]
    p_wd = [p * p_we_factor for p in p_wd]

    # from here like PyCity Alias

    # time series for return statement
    water = []  # in L/h

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

        # Include current_water and current_heat in water and heat
        water.extend(water_daily)

    water_LperH = water

    # Plot
    dhw_demand = compute_stats_and_plot_demand(
        method='DHWcalc_Alias',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )

    return dhw_demand, water_LperH


def generate_daily_probability_step_function(mode, s_step):

    # probability for day with 6 periods. Each Day starts at 0:00. Steps in
    # hours. Sum of steps has to be 24. Sum of probabilites has to be 1.

    if mode == 'weekday':
        step_0 = 6.5
        p_0 = 0.01

        step_1 = 1
        p_1 = 0.5

        step_2 = 4.5
        p_2 = 0.06

        step_3 = 1
        p_3 = 0.16

        step_4 = 5
        p_4 = 0.06

        step_5 = 4
        p_5 = 0.2

        step_6 = 2
        p_6 = 0.01

    elif mode == 'weekend':
        step_0 = 7
        p_0 = 0.02

        step_1 = 2
        p_1 = 0.475

        step_2 = 6
        p_2 = 0.071

        step_3 = 2
        p_3 = 0.237

        step_4 = 3
        p_4 = 0.036

        step_5 = 3
        p_5 = 0.143

        step_6 = 1
        p_6 = 0.018

    else:
        raise Exception('Unkown Mode. Please Choose "Weekday" or "Weekend".')

    steps_wd = [step_0, step_1, step_2, step_3, step_4, step_5, step_6]
    ps_wd = [p_0, p_1, p_2, p_3, p_4, p_5, p_6]

    assert sum(steps_wd) == 24
    assert sum(ps_wd) == 1

    p_0_lst = [p_0 for i in range(int(step_0 * 3600 / s_step))]
    p_1_lst = [p_1 for i in range(int(step_1 * 3600 / s_step))]
    p_2_lst = [p_2 for i in range(int(step_2 * 3600 / s_step))]
    p_3_lst = [p_3 for i in range(int(step_3 * 3600 / s_step))]
    p_4_lst = [p_4 for i in range(int(step_4 * 3600 / s_step))]
    p_5_lst = [p_5 for i in range(int(step_5 * 3600 / s_step))]
    p_6_lst = [p_6 for i in range(int(step_6 * 3600 / s_step))]

    ps_lsts = [p_0_lst, p_1_lst, p_2_lst, p_3_lst, p_4_lst,
               p_5_lst, p_6_lst]

    p_day = []

    for lst in ps_lsts:
        p_day.extend(lst)

    # check if length of daily intervals fits into the stepwidth. if s_step
    # f.e is 3600s (1h), one daily intervall cant be 4.5 hours.
    assert len(p_day) == 24 * 3600 / s_step

    return p_day


def generate_dhw_profile_pycity_alias(s_step=60, initial_day=0,
                                      current_occupancy=5, temp_dT=35,
                                      print_stats=True, plot_demand=True,
                                      start_plot='2019-08-01',
                                      end_plot='2019-08-03', save_fig=False):
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

        # Include current_water and current_heat in water and heat
        water.extend(water_daily)

    water_LperH = water

    # Plot
    dhw_demand = compute_stats_and_plot_demand(
        method='DHWcalc_Alias',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )



def generate_dhw_profile_pycity(s_step=60, temp_dT=35, print_stats=True,
                                plot_demand=True, start_plot='2019-08-01',
                                end_plot='2019-08-03', save_fig=False):
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

    # Plot
    dhw_demand2 = compute_stats_and_plot_demand(
        method='DHWcalc_Alias',
        s_step=s_step,
        water_LperH=water_LperH,
        start_plot=start_plot,
        end_plot=end_plot,
        temp_dT=temp_dT,
        print_stats=print_stats,
        plot_demand=plot_demand,
        save_fig=save_fig
    )

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


def compute_stats_and_plot_demand(method, s_step, water_LperH,
                                  plot_demand=True, start_plot='2019-02-01',
                                  end_plot='2019-02-05', temp_dT=35,
                                  print_stats=False, save_fig=False):
    """
    Takes a timeseries of waterflows per timestep in [L/h]. Computes a
    DHW Demand series in [kWh]. Computes additional stats an optionally
    prints them out. Optionally plots the timesieries with additional stats.

    :param method:      str:    Name of the DHW Method, f.e. DHWcalc, PyCity.
                                Just for naming the plot.
    :param s_step:      int:    seconds within a timestep. F.e. 60, 600, 3600
    :param water_LperH: list:   list that holds the waterflow values for each
                                timestep in Liters per Hour.
    :param start_plot:  str:    start date of the plot. F.e. 2019-01-01
    :param end_plot:    str:    end date of the plot. F.e. 2019-02-01
    :param temp_dT:     int:    temperature difference between freshwater and
                                average DHW outlet temperature. F.e. 35
    :param save_fig:    bool:   decide to save plots as pdf

    :return:    fig:    fig:    figure of the plot
                dhw:    list:   list of the heat demand for DHW for each
                                timestep in kWh.
    """

    water_LperSec = [x / 3600 for x in water_LperH]  # L/s each step

    rho = 980 / 1000  # kg/L for Water (at 60°C? at 10°C its = 1)
    cp = 4180  # J/kgK
    dhw = [i * rho * cp * temp_dT for i in water_LperSec]  # in W

    # compute Sums and Maxima for Water and Heat
    yearly_water_demand = round(sum(water_LperSec) * s_step, 1)  # in L
    av_daily_water = [yearly_water_demand / 365 for i in water_LperH]  # L/day
    max_water_flow = round(max(water_LperH), 1)  # in L/h
    yearly_dhw_demand = round(sum(dhw) * s_step / (3600 * 1000),  1)  # kWh
    max_dhw_heat_flow = round(max(dhw) / 1000, 1)  # in kW

    if print_stats:

        print("Yearly drinking water demand from DHWcalc Alias is {:.2f} L"
              " with a maximum of {:.2f} L/h".format(yearly_water_demand,
                                                     max_water_flow))

        print("Yearly DHW energy demand from DHWcalc Alias is {:.2f} kWh"
              " with a maximum of {:.2f} kW".format(yearly_dhw_demand,
                                                    max_dhw_heat_flow))

    if plot_demand:
        # RWTH colours
        rwth_blue = "#00549F"
        rwth_red = "#CC071E"

        # sns.set_style("white")
        sns.set_context("paper")

        # set date range to simplify plot slicing
        date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                                   freq=str(s_step) + 'S')
        date_range = date_range[:-1]

        # make dataframe for plotting with seaborn
        plot_df = pd.DataFrame({'Waterflow [L/h]': water_LperH,
                                'Yearly av. Demand [L/day]': av_daily_water},
                               index=date_range)

        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        plt.legend()

        plt.title('Water and Heat time-series from {}, dT = {} °C, '
                  'timestep = {} s\n'
                  'Yearly Water Demand = {} L with a Peak of {} L/h \n'
                  'Yearly Heat Demand = {} kWh with a Peak of {} kW'.format(
            method, temp_dT, s_step, yearly_water_demand, max_water_flow,
            yearly_dhw_demand, max_dhw_heat_flow))

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
            fig.savefig(dir_output / "Demand_{}_sliced.pdf".format(method))

    return dhw  # in kWh


if __name__ == '__main__':
    main()


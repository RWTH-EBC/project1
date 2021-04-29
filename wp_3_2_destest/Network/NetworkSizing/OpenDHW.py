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

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=True,
        normalize_mode='sum',
        s_step=60
    )

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=True,
        normalize_mode='max',
        s_step=60
    )

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=False,
        s_step=60
    )

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=True,
        normalize_mode='max',
        s_step=600
    )

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=True,
        normalize_mode='sum',
        s_step=6000
    )

    generate_dhw_profile_dhwcalc_alias(
        normalize_probability_distribution=False,
        s_step=600
    )


    #
    # x, water_pycity_alias_60 = generate_dhw_profile_pycity_alias(s_step=60)
    # x, water_pycity_alias_600 = generate_dhw_profile_pycity_alias(s_step=600)
    #
    # x, water_pycity_60 = generate_dhw_profile_pycity(s_step=60)
    # x, water_pycity_600 = generate_dhw_profile_pycity(s_step=600)

    # x, water_dhwcalc_60 = import_from_dhwcalc(s_step=60)
    # x, water_dhwcalc_600 = import_from_dhwcalc(s_step=600)

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


def shift_weekend_weekday(p_weekday, p_weekend, factor=1.2):
    """
    Shifts the probabilities between the weekday list and the weekend list by a
    defined factor. If the factor is bigger than 1, the probability on the
    weekend is increased. If its smaller than 1, the probability on the
    weekend is decreased.

    :param p_weekday:   list:   probabilites for 1 day of the week [0...1]
    :param p_weekend:   list:   probabilitiers for 1 day of the weekend [0...1]
    :param factor:      float:  factor to shift the probabiliters between
                                weekdays and weekenddays
    :return:
    """

    av_p_wd = statistics.mean(p_weekday)
    av_p_we = statistics.mean(p_weekend)

    av_p_week = av_p_wd * 5 / 7 + av_p_we * 2 / 7

    p_wd_factor = 1 / (5 / 7 + factor * 2 / 7)
    p_we_factor = 1 / (1 / factor * 5 / 7 + 2 / 7)

    assert p_wd_factor * 5 / 7 + p_we_factor * 2 / 7 == 1

    p_wd_weighted = [p * p_we_factor for p in p_weekday]
    p_we_weighted = [p * p_we_factor for p in p_weekend]

    av_p_wd_weighted = statistics.mean(p_wd_weighted)
    av_p_we_weighted = statistics.mean(p_we_weighted)

    av_p_week_weighted = av_p_wd_weighted * 5 / 7 + av_p_we_weighted * 2 / 7

    return p_wd_weighted, p_we_weighted, av_p_week_weighted


def generate_average_daily_profile(mode, l_day, sigma_day, av_p_day,
                                   s_step, plot_profile=False):
    """
    Generates an average profile for daily water drawoffs. The total amount
    of water in the average profile has to be higher than the demanded water
    per day, as the average profile is multiplied by the average probability
    each day. two modes are given to generate the average profile.

    :param mode:            string: type of probability distribution
    :param l_day:           float:  mean value of resulting profile
    :param sigma_day:       float:  standard deviation of resulting profile
    :param av_p_day:        float:  average probability of
    :param s_step:          int:    seconds within a time step
    :param plot_profile:    bool:   decide to plot the profile

    :return: average_profile:   list:   average water drawoff profile in L/h
                                        per timestep
    """

    timesteps_day = int(24 * 3600 / s_step)

    l_av_profile = l_day / av_p_day
    sigma_av_profile = sigma_day / av_p_day

    LperH_step_av_profile = l_av_profile / 24
    sigma_step_av_profile = sigma_av_profile / 24

    if mode == 'gauss':

        # problem: generates negative values.

        average_profile = [random.gauss(LperH_step_av_profile,
                                        sigma=sigma_step_av_profile) for i in
                           range(timesteps_day)]

        if min(average_profile) < 0:
            raise Exception("negative values in average profiles detected. "
                            "Choose a different mean or standard deviation, "
                            "or choose a differnt mode to create the average "
                            "profile.")

    elif mode == 'gauss_abs':

        # If we take the absolute of the gauss distribution, we have no more
        # negative values, but the mean and standard deviation changes,
        # and more than 200 L/d are being consumed.

        average_profile = [random.gauss(LperH_step_av_profile,
                                        sigma=sigma_step_av_profile) for i in
                           range(timesteps_day)]

        average_profile_abs = [abs(entry) for entry in average_profile]

        if statistics.mean(average_profile) != statistics.mean(
                average_profile_abs):

            scale = statistics.mean(average_profile) / statistics.mean(
                average_profile_abs)

            average_profile = [i * scale for i in average_profile_abs]

    elif mode == 'lognormal':

        # problem: understand the settings of the lognormal function.
        # https://en.wikipedia.org/wiki/Log-normal_distribution

        m = LperH_step_av_profile
        sigma = sigma_step_av_profile / 40

        v = sigma ** 2
        norm_mu = np.log(m ** 2 / np.sqrt(v + m ** 2))
        norm_sigma = np.sqrt((v / m ** 2) + 1)

        average_profile = np.random.lognormal(norm_mu, norm_sigma,
                                              timesteps_day)

    else:
        raise Exception("Unkown Mode for average daily water profile "
                        "geneartion")

    if plot_profile:
        mean = [statistics.mean(average_profile) for i in average_profile]
        plt.plot(average_profile)
        plt.plot(mean)
        plt.show()

    return average_profile


def generate_dhw_profile_dhwcalc_alias(normalize_probability_distribution,
                                       s_step, normalize_mode='max',
                                       initial_day=0, temp_dT=35,
                                       print_stats=True, plot_demand=True,
                                       start_plot='2019-01-01',
                                       end_plot='2019-01-03', save_fig=False):

    timesteps_day = int(24 * 3600 / s_step)

    p_we = generate_daily_probability_step_function(
        mode='weekend',
        s_step=s_step
    )

    p_wd = generate_daily_probability_step_function(
        mode='weekday',
        s_step=s_step
    )

    p_wd_weighted, p_we_weighted, av_p_week_weighted = shift_weekend_weekday(
        p_weekday=p_wd,
        p_weekend=p_we,
        factor=1.2
    )

    average_profile = generate_average_daily_profile(
        mode='gauss_abs',
        l_day=200,
        sigma_day=70,
        av_p_day=av_p_week_weighted,
        s_step=s_step,
    )

    # time series for return statement
    water_LperH = []  # in L/h
    p_final = []

    if not normalize_probability_distribution:

        # how it is done in PyCity

        for day in range(365):

            # Is the current day on a weekend?
            if (day + initial_day) % 7 >= 5:
                p_day = p_we_weighted
            else:
                p_day = p_wd_weighted

            water_daily = []

            # Compute seasonal factor
            arg = math.pi * (2 / 365 * day - 1 / 4)
            probability_season = 1 + 0.1 * np.cos(arg)

            for step in range(timesteps_day):

                # Compute probability for tap water demand at timestep
                probability = p_day[step] * probability_season
                p_final.append(probability)

                # Check if tap water demand occurs. The higher the probability,
                # the more likely the if statement is true.
                if random.random() < probability:

                    # Compute amount of tap water consumption. Start with seed?
                    # This consumption has to be positive!
                    # Problem: when the absolute is taken, the mean and sigma of
                    # the gauss distribution is changed, and the total daily
                    # draw-off volume changes!
                    water_t = random.gauss(average_profile[step], sigma=114.33)
                    water_daily.append(abs(water_t))
                else:
                    water_daily.append(0)

            # Include current_water and current_heat in water and heat
            water_LperH.extend(water_daily)

    else:   # normalize probability distribution

        for day in range(365):

            # Is the current day on a weekend?
            if (day + initial_day) % 7 >= 5:
                p_day = p_we_weighted
            else:
                p_day = p_wd_weighted

            # Compute seasonal factor
            arg = math.pi * (2 / 365 * day - 1 / 4)
            probability_season = 1 + 0.1 * np.cos(arg)

            for step in range(timesteps_day):

                probability = p_day[step] * probability_season
                p_final.append(probability)

        if normalize_mode == 'sum':     # how it is probably done in DHWcalc

            p_norm_integral = normalize_and_sum_list(lst=p_final)

            drawoffs, p_drawoffs = generate_drawoffs(
                mean_vol_per_drawoff=8,
                mean_drawoff_vol_per_day=200,
                s_step=s_step,
                p_norm_integral=p_norm_integral
            )

            water_LperH = distribute_drawoffs(
                drawoffs=drawoffs,
                p_drawoffs=p_drawoffs,
                p_norm_integral=p_norm_integral,
                s_step=s_step
            )

            # drawoffs_shower, p_drawoffs = generate_drawoffs(
            #     mean_vol_per_drawoff=8,
            #     mean_drawoff_vol_per_day=25,
            #     s_step=s_step,
            #     p_norm_integral=p_norm_integral
            # )
            #
            # water_LperH = distribute_drawoffs(
            #     drawoffs=drawoffs,
            #     p_drawoffs=p_drawoffs,
            #     p_norm_integral=p_norm_integral,
            #     s_step=s_step
            # )



        elif normalize_mode == 'max':

            # PyCity algorythm, probabilities are scaled to their max

            max_p_final = max(p_final)
            p_norm = [float(i) / max_p_final for i in p_final]

            average_profile = average_profile * 365

            for step in range(365 * timesteps_day):

                if random.random() < p_norm[step]:

                    water_t = random.gauss(average_profile[step], sigma=114.33)
                    water_LperH.append(abs(water_t))
                else:
                    water_LperH.append(0)

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


def distribute_drawoffs(drawoffs, p_drawoffs, p_norm_integral, s_step):
    """
    Takes a small list (p_drawoffs) and sorts it into a bigger list (
    p_norm_integral). Both lists are being sorted. Then, the big list is
    iterated over, and whenever a value of the small list is smaller than a
    value of the big list, the index of the big list is saved and a drawoff
    event from the drawoffs list occurs.

    :param drawoffs:        list:   drawoff events in L/h
    :param p_drawoffs:      list:   drawoff event probabilities [0...1]
    :param p_norm_integral: list:   normalized sum of yearly water use
                                    probabilities [0...1]
    :param s_step:          int:    seconds within a timestep

    :return: water_LperH:   list:   resutling water drawoff profile
    """

    p_drawoffs.sort()
    p_norm_integral.sort()

    drawoff_count = 0

    # for return statement
    water_LperH = [0] * int(365 * 24 * 3600 / s_step)

    for step, p_current_sum in enumerate(p_norm_integral):

        if p_drawoffs[drawoff_count] < p_current_sum:
            water_LperH[step] = drawoffs[drawoff_count]
            drawoff_count += 1

            if drawoff_count >= len(drawoffs):
                break

    return water_LperH


def generate_drawoffs(mean_vol_per_drawoff, mean_drawoff_vol_per_day,
                      s_step, p_norm_integral):

    # dhw calc has more settings here, see Fig 5 in paper "Draw off features".

    av_drawoff_flow_rate = mean_vol_per_drawoff * 3600 / s_step     # in L/h

    mean_no_drawoffs_per_day = mean_drawoff_vol_per_day / mean_vol_per_drawoff

    total_drawoffs = int(mean_no_drawoffs_per_day * 365)

    mu = av_drawoff_flow_rate
    sig = av_drawoff_flow_rate / 6
    drawoffs = [random.gauss(mu, sigma=sig) for i in range(total_drawoffs)]

    # drawoff flow rate has to be positive. maybe reduce standard deviation.
    assert min(drawoffs) >= 0

    min_rand = min(p_norm_integral)
    max_rand = max(p_norm_integral)

    p_drawoffs = [random.uniform(min_rand, max_rand) for i in drawoffs]
    p_drawoffs.sort()

    return drawoffs, p_drawoffs


def normalize_and_sum_list(lst):
    """
    takes a list and normalizes it based on the sum of all list elements.
    then generates a new list based on the current sum of each list entry.

    :param lst:                 list:   input list
    :return: lst_norm_integral: list    output list
    """

    sum_lst = sum(lst)
    lst_norm = [float(i) / sum_lst for i in lst]

    current_sum = 0
    lst_norm_integral = []

    for entry in lst_norm:
        current_sum += entry
        lst_norm_integral.append(current_sum)

    return lst_norm_integral


def generate_daily_probability_step_function(mode, s_step, plot_p_day=False):
    """
    Generates probabilites for a day with 6 periods. Corresponds to the mode
    "step function for weekdays and weekends" in DHWcalc and uses the same
    standard values. Each Day starts at 0:00. Steps in hours. Sum of steps
    has to be 24. Sum of probabilites has to be 1.

    :param mode:        string: decide to compute for a weekday of a weekend day
    :param s_step:      int:    seconds within a timestep
    :param plot_p_day:  Bool:   decide to plot the probability distribution
    :return: p_day      list:   the probability distribution for one day.
    """

    if mode == 'weekday':

        steps_and_ps = [(6.5, 0.01), (1, 0.5), (4.5, 0.06), (1, 0.16),
                        (5, 0.06), (4, 0.2), (2, 0.01)]

    elif mode == 'weekend':

        steps_and_ps = [(7, 0.02), (2, 0.475), (6, 0.071), (2, 0.237),
                        (3, 0.036), (3, 0.143), (1, 0.018)]

    else:
        raise Exception('Unkown Mode. Please Choose "Weekday" or "Weekend".')

    steps = [tup[0] for tup in steps_and_ps]
    ps = [tup[1] for tup in steps_and_ps]

    assert sum(steps) == 24
    assert sum(ps) == 1

    p_day = []

    for tup in steps_and_ps:
        p_lst = [tup[1] for i in range(int(tup[0] * 3600 / s_step))]
        p_day.extend(p_lst)

    # check if length of daily intervals fits into the stepwidth. if s_step
    # f.e is 3600s (1h), one daily intervall cant be 4.5 hours.
    assert len(p_day) == 24 * 3600 / s_step

    if plot_p_day:
        plt.plot(p_day)
        plt.show()

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
    av_daily_water = round(yearly_water_demand / 365, 1)
    av_daily_water_lst = [av_daily_water for i in water_LperH]  # L/day
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
                                'Yearly av. Demand [{} L/day]'.format(
                                    av_daily_water): av_daily_water_lst},
                               index=date_range)

        # fig, (ax1, ax2) = plt.subplots(2, 1)
        fig, ax1 = plt.subplots()
        fig.tight_layout()

        ax1 = sns.lineplot(ax=ax1, data=plot_df[start_plot:end_plot],
                           linewidth=1.0, palette=[rwth_blue, rwth_red])

        # ax2 = sns.lineplot(ax=ax2, data=plot_df,
        #                    linewidth=1.0, palette=[rwth_blue, rwth_red])

        plt.legend(loc="upper left")

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


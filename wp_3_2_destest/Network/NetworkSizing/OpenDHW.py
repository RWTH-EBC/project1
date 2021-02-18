# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xlrd
import math
import random
import matplotlib.dates as mdates

import pycity_base.classes.demand.domestic_hot_water as dhw
import pycity_base.classes.timer as time
import pycity_base.classes.weather as weath
import pycity_base.classes.prices as price
import pycity_base.classes.environment as env
import pycity_base.classes.demand.occupancy as occ


def main():

    # generate_dhw_profile_like_pycity()
    import_from_dhwcalc()

    pass


def import_from_dhwcalc(s_step=60, delta_t_dhw=35,
                        plot_demand=True, start_in_summer=False,
                        plot_demand_seaborn=True, save_fig=True):
    """
    DHWcalc yields Volume Flow TimeSeries (in Liters per hour). To get
    Energyflows -> Q = Vdot * rho * cp * dT

    :return: dhw_demand:    time series. each timestep contains the Energyflow
                            in Watt -> W
    """

    if s_step == 60:
        dhw_file = "DHWcalc_200L_1min_1cat_step_functions_DHW.txt"
    elif s_step == 600:
        dhw_file = "DHWcalc_200L_10min_1cat_step_functions_DHW.txt"
    else:
        raise Exception("Unkown Time Step for DHWcalc")

    dhw_profile = Path.cwd() / dhw_file
    dir_output = Path.cwd() / "plots"
    dir_output.mkdir(exist_ok=True)

    water_LperH = [int(word.strip('\n')) for word in
                   open(dhw_profile).readlines()]  # L/h each step
    water_LperSec = [x / 3600 for x in water_LperH]

    rho = 1  # 1L = 1kg for Water
    cp = 4180  # J/kgK
    dhw_demand = [i * rho * cp * delta_t_dhw for i in water_LperSec]  # in W

    yearly_dhw_demand = sum(dhw_demand) * s_step / (3600 * 1000)  # in kWh
    max_dhw_heat_flow = max(dhw_demand) / 1000  # in kW
    print("Yearly DHW energy demand from DHWcalc is {:.2f} kWh"
          " with an average of {:.2f} kW".format(yearly_dhw_demand,
                                                 max_dhw_heat_flow))

    if plot_demand:

        sns.set_style("white")
        sns.set_context("paper")

        fig, ax = plt.subplots()
        ax.plot(dhw_demand, linewidth=0.7, label="DHW Demand in Watt")
        plt.ylabel('DHW demand in Watt')
        plt.xlabel('Minutes in a Year')
        plt.title('Total Energy: {:.2f} kWh with a peak of {:.2f} kW'.format(
            yearly_dhw_demand, max_dhw_heat_flow))
        plt.show()
        if save_fig:
            fig.savefig(dir_output / "DHW_Demand.pdf")

    if plot_demand_seaborn:

        # RWTH colours
        rwth_blue = "#00549F"
        # rwth_red = "#CC071E"

        sns.set_style("white")
        sns.set_context("paper")

        # set date range to simplify plot slicing
        date_range = pd.date_range(start='2019-01-01', end='2020-01-01',
                                   freq=str(s_step) + 'S')
        date_range = date_range[:-1]

        # convert demands to kW for plotting
        dhw_demand = [dem_step / 1000 for dem_step in dhw_demand]

        # make dataframe for plotting with seaborn
        dhw_demand_df = pd.DataFrame({'DHW Demand': dhw_demand},
                                     index=date_range)

        fig, ax = plt.subplots()
        ax_data = dhw_demand_df[['DHW Demand']]
        ax = sns.lineplot(data=ax_data, linewidth=0.7, dashes=False,
                          palette=[rwth_blue])
        ax.legend_.remove()

        # set the x axis ticks
        # https://matplotlib.org/3.1.1/gallery/ticks_and_spines/date_concise_formatter.html
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        formatter.formats = ['%y', '%b', '%d', '%H:%M', '%H:%M', '%S.%f', ]
        formatter.zero_formats = [''] + formatter.formats[:-1]
        formatter.zero_formats[3] = '%d-%b'
        formatter.offset_formats = ['', '%Y', '%b %Y', '%d %b %Y', '%d %b %Y',
                                    '%d %b %Y %H:%M', ]
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        plt.ylabel('DHW demand in kW')

        plt.show()
        if save_fig:
            fig.savefig(dir_output / "DHW_Demand_sns.pdf")

    if start_in_summer:
        half = int((len(
            dhw_demand) - 1) / 2)  # half the length of the demand timeseries
        dhw_demand = dhw_demand[half:] + dhw_demand[
                                         :half]  # shift demand by half a year

    return dhw_demand  # in W


def generate_dhw_profile_like_pycity(s_step=60, initial_day=0,
                                     current_occupancy=5, temp_dT=35,
                                     plot_demand=True, save_fig=False):
    """
    :param: s_step: int:        seconds within a time step. F.e. should be
                                60s for pycity.
    :param: initial_day:        0: Mon, 1: Tue, 2: Wed, 3: Thur,
                                4: Fri, 5 : Sat, 6 : Sun
    :param: current_occuapncy:  number of people in the house. In PyCity,
                                this is a list and the occuancy changes during
                                the year. Between 0 and 5. Values have to be
                                integers.
    :param: temp_dT: int/float: How much does the tap water has to be heated up?
    :return: water: List:       Tap water volume flow in liters per hour.
             heat : List:       Resulting minute-wise sampled heat demand in
                                Watt. The heat capacity of water is assumed
                                to be 4180 J/(kg.K) and the density is
                                assumed to be 980 kg/m3.
    """

    # get dhw stochastical file should be in the same dir as this script.
    src_path = os.path.dirname(__file__)
    profiles_path = os.path.join(src_path, 'dhw_stochastical.xlsx')
    profiles = {"we": {}, "wd": {}}
    book = xlrd.open_workbook(profiles_path)

    # Iterate over all sheets. wd = weekday, we = weekend. mw = ist the
    # average profile. occupancy is between 1-6 (we1 - we6). occupancy of 6
    # people seems to be wrongly implemented in PyCity, as the sum of the
    # probabilities increases with occupancy (1-5) but then decreases for the
    # 6th person.
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

        steps = int(24 * 3600 / s_step)  # should be 1minute steps for pycity
        for t in range(steps):  # Iterate over all time-steps in a day

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
                water_m = random.gauss(average_profile[t], sigma=114.33)
                water_daily.append(abs(water_m))
            else:
                water_daily.append(0)

        c = 4180  # J/(kg.K)
        rho = 0.980  # kg/l
        heat_daily = [i * rho * c * temp_dT / s_step for i in water_daily]  # W

        # Include current_water and current_heat in water and heat
        water[day * 1440:(day + 1) * 1440] = water_daily  # L/h, 1min s_step
        heat[day * 1440:(day + 1) * 1440] = heat_daily  # W, 1min s_step

    # print total energy and average energy flow
    yearly_heat_demand = sum(heat) / 1000  # in kWh
    average_heat_demand = (sum(heat) / len(heat)) / 1000  # in kW per step
    max_heat_demand = max(heat) / 1000  # in kW per step

    print("Yearly heating demand for a from DemGen is {:.2f} kWh "
          "with an average of {:.2f} kW and a peak of {:.2f} kW"
          .format(yearly_heat_demand, average_heat_demand, max_heat_demand))

    yearly_water_demand = sum(water) / 1000  # in kWh
    average_water_demand = (sum(water) / len(water)) / 1000  # in kW
    max_water_demand = max(water) / 1000  # in kW per step

    print("Yearly cooling demand for a house from DemGen is {:.2f} kWh "
          "with an average of {:.2f} kW and a peak of {:.2f} kW"
          .format(yearly_water_demand, average_water_demand, max_water_demand))

    if plot_demand:
        sns.set_context("paper")

        fig, ax = plt.subplots()
        ax.plot(water, linewidth=0.7, label="Water")
        ax.plot(heat, linewidth=0.7, label="Heat")
        plt.ylabel('Heat and Cold demand in Watt'.format(yearly_water_demand,
                                                         yearly_heat_demand))
        plt.xlabel('Minutes in a Year')
        plt.title('Heat and Cold Demand from DemGen, \n'
                  'Total Water Demand: {:.0f} L, with a peak of {:.2f} '
                  'kW \n'
                  'Total Heat Demand: {:.0f} kWh  with a peak of {:.2f} kW'
                  .format(yearly_water_demand, max_water_demand,
                          yearly_heat_demand, max_heat_demand))

        plt.show()

        if save_fig:
            dir_output = os.path.dirname(__file__)
            fig.savefig(os.path.join(dir_output + "/DemGenDemands.pdf"))
            fig.savefig(os.path.join(dir_output + "DemGenDemands.png"), dpi=600)

    return heat, water


def generate_dhw_profile_pycity(plot_demand=False):
    """
    from https://github.com/RWTH-EBC/pyCity
    :return:
    """
    #  Generate environment with timer, weather, and prices objects
    timer = time.Timer(time_discretization=3600,  # in seconds
                       timesteps_total=int(365 * 24)
                       )

    weather = weath.Weather(timer=timer)
    prices = price.Prices()
    environment = env.Environment(timer=timer, weather=weather, prices=prices)

    #  Generate occupancy object with stochastic user profile
    occupancy = occ.Occupancy(environment=environment, number_occupants=5)

    dhw_obj = dhw.DomesticHotWater(
        environment=environment,
        t_flow=60,  # DHW output temperature in degree Celsius
        method=2,  # Stochastic dhw profile, Method 1 not working
        supply_temperature=10,  # DHW inlet flow temperature in degree C.
        occupancy=occupancy.occupancy)  # Occupancy profile (600 sec resolution)
    dhw_demand = dhw_obj.loadcurve  # ndarray with 8760 timesteps in Watt

    if plot_demand:
        plt.plot(dhw_demand)
        plt.ylabel('dhw pycity, sum={:.2f}'.format(sum(dhw_demand) / 1000))
        plt.show()

    return dhw_demand


if __name__ == '__main__':
    main()

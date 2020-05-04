# Created November 2018
# Ina De Jaeger

"""This module contains an example how to import TEASER projects from
*.teaserXML and pickle in order to reuse data.
"""
import teaser.data.output.ideas_district_simulation as simulations
import teaser.data.input.citygml_input as citygml_in
import os
import pandas as pd
import numpy as np
from teaser.project import Project
import matplotlib.pyplot as plt
from matplotlib import style
import collections
from modelicares import SimRes

style.use("ggplot")


def plot_ldc(outputDir, dfs, styles, title):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (20, 12),
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    plt.rcParams.update(params)

    # Plot LDC
    print("Creating LDC plot")
    for library, df in dfs.items():
        print(library)
        # print df.shape
        # convert to proper units
        if not isinstance(df.index, pd.DatetimeIndex):
            df["index"] = pd.to_datetime("2018-01-01 00:00") + pd.to_timedelta(
                df.index, unit="s"
            )
            df.set_index(df["index"], inplace=True)
            df.drop(columns=["index"], inplace=True)
        df.sort_values(by=["Qheating_building_W"], ascending=False, inplace=True)
        df["Qheating_building_W"] = df["Qheating_building_W"] / 1000
        if df.shape[0] == 52561:
            # print library + ": sampling time is 600 s"
            x = [time * 600 / 3600.0 for time in range(df.shape[0])]
        elif df.shape[0] == 35041:
            # print library + ": sampling time is 900 s"
            x = [time * 900 / 3600.0 for time in range(df.shape[0])]
        else:
            print(
                "I could not determine the sampling time properly, please have a look yourself."
            )
        plt.plot(
            x,
            df["Qheating_building_W"].values,
            label=library,
            alpha=1,
            color=styles[library][0],
            linestyle=styles[library][1],
            linewidth=1.0,
        )

    plt.xlabel("Duration [h]")
    plt.ylabel("Heating power [kW]")
    plt.xlim(-1, x[-1])
    # Put legend
    plt.legend()
    plt.title(title)
    # plt.tight_layout()
    # plt.show()
    plt.savefig(outputDir + "LDC.png", dpi=300)
    plt.close()


def plot_profiles(outputDir, dfs, styles, title):
    dates = [
        ["March", pd.to_datetime("2018-03-21"), pd.to_datetime("2018-03-28")],
        ["June", pd.to_datetime("2018-06-21"), pd.to_datetime("2018-06-28")],
        ["September", pd.to_datetime("2018-09-21"), pd.to_datetime("2018-09-28")],
        ["December", pd.to_datetime("2018-12-21"), pd.to_datetime("2018-12-28")],
    ]
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (20, 36),
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    plt.rcParams.update(params)

    print("Creating profile plots")
    for date in dates:
        # We want all on one plot
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(20, 12))
        ax1, ax2, ax3 = axes
        # Plot temperatures of day zone and night zone
        for zone in ["dayzone", "nightzone"]:
            print(date[0])
            print(zone)
            for library, df in dfs.items():
                print("     " + str(library))
                # convert to proper units
                if not isinstance(df.index, pd.DatetimeIndex):
                    df["index"] = pd.to_datetime("2018-01-01 00:00") + pd.to_timedelta(
                        df.index, unit="s"
                    )
                    df.set_index(df["index"], inplace=True)
                    df.drop(columns=["index"], inplace=True)
                df = df[(df.index > date[1]) & (df.index < date[2])]
                if zone == "dayzone":
                    ax = ax1
                else:
                    ax = ax2
                df["Tair_" + zone + "_C"].plot.line(
                    label=str(library) + " - Tair",
                    alpha=1,
                    color=styles[library][0],
                    linestyle=styles[library][1],
                    linewidth=1.0,
                    ax=ax,
                )
                ax.set_ylabel("Air temperature of " + zone + " [degC]")
                ax.grid(which="both", axis="both", color="white")

        # Plot heating power
        print("Power")
        for library, df in dfs.items():
            # convert to proper units
            if not isinstance(df.index, pd.DatetimeIndex):
                df["index"] = pd.to_datetime("2018-01-01 00:00") + pd.to_timedelta(
                    df.index, unit="s"
                )
                df.set_index(df["index"], inplace=True)
                df.drop(columns=["index"], inplace=True)
            df = df[(df.index > date[1]) & (df.index < date[2])]
            ax = ax3
            df["Qheating_building_W"].plot.line(
                label=library,
                alpha=1,
                color=styles[library][0],
                linestyle=styles[library][1],
                linewidth=1.0,
                ax=ax,
            )
            ax.set_ylabel("Heating power [W]")
            ax.grid(which="both", axis="both", color="white")

        plt.xlim(date[1], date[2])
        # Put legend
        ax3.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4
        )
        ax1.set_title(
            (outputDir.rsplit("/", 1)[1] + "profiles_" + date[0]).replace("_", " - ")
        )
        # plt.tight_layout()
        # plt.show()
        plt.savefig(outputDir + "profiles_" + date[0] + ".png", dpi=300)
        plt.close()


def calc_numbers(dfs, outputDir, styles, title):
    params = {
        "legend.fontsize": "large",
        "figure.figsize": (7, 5),
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
    }
    plt.rcParams.update(params)
    # Calculate numbers and add to df
    df_numbers = pd.DataFrame()
    df_temps = pd.DataFrame()
    print("Calculating and plotting numbers")
    for library, df in dfs.items():
        print(library)
        ### TEMPERATURE
        try:
            df_numbers.loc["Tair_dayzone_overheating_Kh", library] = calc_overheating(
                df["Tair_dayzone_K"]
            )
            df_numbers.loc["Tair_dayzone_mean_degC", library] = (
                df["Tair_dayzone_K"].mean() - 273.15
            )
            df_numbers.loc["Tair_dayzone_min_degC", library] = (
                df["Tair_dayzone_K"].min() - 273.15
            )
            df_numbers.loc["Tair_dayzone_max_degC", library] = (
                df["Tair_dayzone_K"].max() - 273.15
            )

            df_temps.loc[
                "T_dayzone_overheating_Kh", library + " - Tair"
            ] = calc_overheating(df["Tair_dayzone_K"])
            df_temps.loc["T_dayzone_mean_degC", library + " - Tair"] = (
                df["Tair_dayzone_K"].mean() - 273.15
            )
            df_temps.loc["T_dayzone_min_degC", library + " - Tair"] = (
                df["Tair_dayzone_K"].min() - 273.15
            )
            df_temps.loc["T_dayzone_max_degC", library + " - Tair"] = (
                df["Tair_dayzone_K"].max() - 273.15
            )

        except:
            pass
        try:
            df_numbers.loc[
                "Toperative_dayzone_overheating_Kh", library
            ] = calc_overheating(df["Toperative_dayzone_K"])
            df_numbers.loc["Toperative_dayzone_mean_degC", library] = (
                df["Toperative_dayzone_K"].mean() - 273.15
            )
            df_numbers.loc["Toperative_dayzone_min_degC", library] = (
                df["Toperative_dayzone_K"].min() - 273.15
            )
            df_numbers.loc["Toperative_dayzone_max_degC", library] = (
                df["Toperative_dayzone_K"].max() - 273.15
            )

            df_temps.loc[
                "T_dayzone_overheating_Kh", library + " - Toper"
            ] = calc_overheating(df["Toperative_dayzone_K"])
            df_temps.loc["T_dayzone_mean_degC", library + " - Toper"] = (
                df["Toperative_dayzone_K"].mean() - 273.15
            )
            df_temps.loc["T_dayzone_min_degC", library + " - Toper"] = (
                df["Toperative_dayzone_K"].min() - 273.15
            )
            df_temps.loc["T_dayzone_max_degC", library + " - Toper"] = (
                df["Toperative_dayzone_K"].max() - 273.15
            )
        except:
            pass
        try:
            df_numbers.loc["Tair_nightzone_overheating_Kh", library] = calc_overheating(
                df["Tair_nightzone_K"]
            )
            df_numbers.loc["Tair_nightzone_mean_degC", library] = (
                df["Tair_nightzone_K"].mean() - 273.15
            )
            df_numbers.loc["Tair_nightzone_min_degC", library] = (
                df["Tair_nightzone_K"].min() - 273.15
            )
            df_numbers.loc["Tair_nightzone_max_degC", library] = (
                df["Tair_nightzone_K"].max() - 273.15
            )

            df_temps.loc[
                "T_nightzone_overheating_Kh", library + " - Tair"
            ] = calc_overheating(df["Tair_nightzone_K"])
            df_temps.loc["T_nightzone_mean_degC", library + " - Tair"] = (
                df["Tair_nightzone_K"].mean() - 273.15
            )
            df_temps.loc["T_nightzone_min_degC", library + " - Tair"] = (
                df["Tair_nightzone_K"].min() - 273.15
            )
            df_temps.loc["T_nightzone_max_degC", library + " - Tair"] = (
                df["Tair_nightzone_K"].max() - 273.15
            )
        except:
            pass

        ### POWER
        try:
            df_numbers.loc["Power_building_max_kW", library] = calc_peak_power(
                df["Qheating_building_W"]
            )
            df_numbers.loc["Energydemand_building_kWh", library] = calc_energy_demand(
                df["Qheating_building_W"]
            )
        except:
            pass
        try:
            df_numbers.loc["Power_dayzone_max_W", library] = calc_peak_power(
                df["Qheating_dayzone_W"]
            )
        except:
            pass
        try:
            df_numbers.loc["Power_nightzone_max_W", library] = calc_peak_power(
                df["Qheating_nightzone_W"]
            )
        except:
            pass

    df_numbers.sort_index(inplace=True)
    df_numbers.T.to_csv(outputDir + "summary_allvariables.csv", sep=";")
    df_temps.sort_index(inplace=True)
    df_temps.to_csv(outputDir + "summary_temperatures.csv", sep=";")

    kpi_aliases = {
        "T_dayzone_max_degC": "Maximal",
        "T_dayzone_mean_degC": "Mean ",
        "T_dayzone_min_degC": "Minimal",
        "T_dayzone_overheating_Kh": "Day zone",
        "T_nightzone_max_degC": "Maximal",
        "T_nightzone_mean_degC": "Mean",
        "T_nightzone_min_degC": "Minimal",
        "T_nightzone_overheating_Kh": "Night zone",
    }

    # Plot temperatures on 1 plot
    try:
        params = {
            "legend.fontsize": "large",
            "figure.figsize": (15, 5),
            "axes.labelsize": "large",
            "axes.titlesize": "large",
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
        }
        plt.rcParams.update(params)
        fig, axes = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(20, 12))
        ax1, ax2, ax3 = axes
        # Top: day zone temperatures
        kpis = ["T_dayzone_max_degC", "T_dayzone_mean_degC", "T_dayzone_min_degC"]
        df_temps.loc[kpis, :].plot.bar(
            color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns],
            ax=ax1,
            legend=False,
        )
        for p in ax1.patches:
            ax1.annotate(
                str(round(p.get_height(), 1)),
                (p.get_x() * 1.005, p.get_height() * 1.005 + 2.5),
                rotation=90,
            )
        ax1.set_xticklabels([kpi_aliases[i] for i in kpis], rotation=0, ha="center")
        ax1.set_ylim([0, 40])
        ax1.set_ylabel("Air temperature of day zone [degC]")
        ax1.set_title(title)
        # Middle: night zone temperatures
        kpis = ["T_nightzone_max_degC", "T_nightzone_mean_degC", "T_nightzone_min_degC"]
        df_temps.loc[kpis, :].plot.bar(
            color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns],
            ax=ax2,
            legend=False,
        )
        for p in ax2.patches:
            ax2.annotate(
                str(round(p.get_height(), 1)),
                (p.get_x() * 1.005, p.get_height() * 1.005 + 2.5),
                rotation=90,
            )
        ax2.set_xticklabels([kpi_aliases[i] for i in kpis], rotation=0, ha="center")
        ax2.set_ylim([0, 40])
        ax2.set_ylabel("Air temperature of night zone [degC]")
        # Bottom: overheating
        kpis = ["T_dayzone_overheating_Kh", "T_nightzone_overheating_Kh"]
        df_temps.loc[kpis, :].plot.bar(
            color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns],
            ax=ax3,
            legend=False,
        )
        """for p in ax3.patches:
			ax3.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.02, p.get_height() * 1.005 + 600), rotation=90)#"""
        ax3.set_xticklabels([kpi_aliases[i] for i in kpis], rotation=0, ha="center")
        # ax3.set_ylim([0, 5500])
        ax3.set_ylabel("Overheating [Kh]")
        ax3.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4
        )
        plt.tight_layout()
        plt.savefig(outputDir + "temperatures.png", dpi=300)
        plt.close()
    except:
        plt.close()

    # Plot peak power and annual energy demand on 1 plot
    try:
        params = {
            "legend.fontsize": "large",
            "figure.figsize": (15, 5),
            "axes.labelsize": "large",
            "axes.titlesize": "large",
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
        }
        plt.rcParams.update(params)
        fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(20, 12))
        ax1, ax2 = axes
        # Left: peak power
        df_numbers.loc["Power_building_max_kW", :].plot.bar(
            rot=0,
            color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns],
            ax=ax1,
            legend=False,
        )
        ax1.set_xticklabels(df_numbers.columns, rotation=40, ha="right")
        for p in ax1.patches:
            ax1.annotate(
                str(round(p.get_height(), 1)),
                (p.get_x() * 1.005, p.get_height() * 1.005),
            )
        ax1.set_ylabel("Peak power [kW]")
        # Right: annual energy demand for space heating
        df_numbers.loc["Energydemand_building_kWh", :].plot.bar(
            rot=0,
            color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns],
            ax=ax2,
            legend=False,
        )
        ax2.set_xticklabels(df_numbers.columns, rotation=40, ha="right")
        for p in ax2.patches:
            ax2.annotate(
                str(round(p.get_height(), 1)),
                (p.get_x() * 1.005, p.get_height() * 1.005),
            )
        ax2.set_ylabel("Annual energy demand [kWh]")
        plt.tight_layout()
        plt.savefig(outputDir + "powers.png", dpi=300)
        plt.close()
    except:
        plt.close()

    """try:
		kpis = ['T_nightzone_max_degC', 'T_nightzone_mean_degC', 'T_nightzone_min_degC']
		ax = df_temps.loc[kpis, :].plot.bar(color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns])
		for p in ax.patches:
			ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005 + 2.5), rotation=90)
		ax.set_xticklabels([kpi_aliases[i] for i in kpis], rotation=0, ha='center')
		ax.set_ylim([0, 40])
		plt.legend(loc='upper right')
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=False, ncol=4)
		plt.ylabel("Air temperature of night zone [degC]")
		plt.title(title)
		plt.tight_layout()
		plt.savefig(outputDir + "night_temps.png", dpi=300)
		plt.close()
	except:
		pass

	try:
		params = {'legend.fontsize': 'large',
				  'figure.figsize': (12, 5),
				  'axes.labelsize': 'large',
				  'axes.titlesize': 'large',
				  'xtick.labelsize': 'large',
				  'ytick.labelsize': 'large'}
		plt.rcParams.update(params)
		kpis = ['T_dayzone_overheating_Kh', 'T_nightzone_overheating_Kh']
		ax = df_temps.loc[kpis, :].plot.bar(color=[styles[i.split(" - ")[0]][0] for i in df_temps.columns])
		for p in ax.patches:
			ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.02, p.get_height() * 1.005 + 600), rotation=90)
		ax.set_xticklabels([kpi_aliases[i] for i in kpis], rotation=0, ha='center')
		ax.set_ylim([0, 5500])
		plt.legend(loc='upper right')
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), fancybox=True, shadow=False, ncol=4)
		plt.ylabel("Overheating [Kh]")
		plt.title(title)
		plt.tight_layout()
		plt.savefig(outputDir + "overheating.png", dpi=300)
		plt.close()
	except:
		pass

	try:
		# I prefer to plot all temperatures together, so re-order df_number
		params = {'legend.fontsize': 'large',
				  'figure.figsize': (7, 5),
				  'axes.labelsize': 'large',
				  'axes.titlesize': 'large',
				  'xtick.labelsize': 'large',
				  'ytick.labelsize': 'large'}
		plt.rcParams.update(params)
		for var in df_temps.index:
			df_temps.loc[var, :].plot.bar(rot=45, color=[styles[i.split(' - ')[0]][0] for i in df_temps.columns])
			ax = plt.gca()
			ax.set_xticklabels(df_numbers.columns, rotation=40, ha='right')
			for p in ax.patches:
				ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))
			plt.ylabel(var)
			plt.title(title)
			plt.tight_layout()
			# plt.show()
			plt.savefig(outputDir + var + ".png", dpi=300)
			plt.close()
	except:
		pass


	# Plot variables separately
	for var in df_numbers.index:
		if "Tair" in var:
			df_numbers.loc[var, :].plot.bar(rot=0, color=[styles[i][0] for i in df_numbers.columns])
			ax = plt.gca()
			ax.set_xticklabels(df_numbers.columns, rotation=40, ha='right')
			for p in ax.patches:
				ax.annotate(str(round(p.get_height(), 1)), (p.get_x() * 1.005, p.get_height() * 1.005))
			plt.ylabel(var)
			plt.title(title)
			plt.tight_layout()
			# plt.show()
			plt.savefig(outputDir + var + ".png", dpi=300)
			plt.close()
			print(var)
		else:
			df_numbers.loc[var, :].plot.bar(rot=45, color=[styles[i][0] for i in df_numbers.columns])
			ax = plt.gca()
			ax.set_xticklabels(df_numbers.columns, rotation=40, ha='right')
			for p in ax.patches:
				ax.annotate(str(int(round(p.get_height()))), (p.get_x() * 1.005, p.get_height() * 1.005))
			plt.ylabel(var)
			plt.title(title)
			plt.tight_layout()
			# plt.show()
			plt.savefig(outputDir + var + ".png", dpi=300)
			plt.close()#"""


def calc_overheating(df):  # =df["TSensor of " + zone_name + " / K"]
    # calculate overheating of zone [Ks] (over 25degC = 298.15 K)
    timearray = df.index.values
    temparray = df.values
    overheatingarray = [
        (temp - 298.15) for temp in temparray
    ]  # this array contains all temp - 25degC, if negative, set to 0, then integrate
    for index, temp in enumerate(overheatingarray):
        if temp < 0.0:
            overheatingarray[index] = 0.0
    overheating = np.trapz(y=overheatingarray, x=timearray)
    # calculate overheating of zone [Kh]
    overheating = overheating / 3600

    return overheating


def calc_energy_demand(df):  # =df['Qheating_building_W']
    # Calculate building-related KPI (= Q heating system building)
    # calculate energy use for space heating [J = Ws]
    timearray = df.index.values
    qarray = df.values
    energyuse = np.trapz(y=qarray, x=timearray)
    # calculate energy use for space heating [kWh]
    energyuse = energyuse / 1000 / 3600
    print(energyuse)
    return energyuse


def calc_peak_power(df):  # =df['Q heating system building / W']
    # calculate peak power building [W]
    peakpower = df.max()
    # calculate peak power building [kW]
    peakpower = peakpower / 1000

    return peakpower


def import_results(inputDir):
    # Import results just once

    dfs = collections.OrderedDict()
    fileNames = os.listdir(inputDir)
    print(fileNames)
    for fileName in fileNames:
        df = pd.read_csv(inputDir + fileName, sep=",", index_col=0, header=0)
        print(df.head())
        dfs.update([(fileName[:-4], df)])

    variables_to_C = ["Tair_dayzone_K", "Tair_nightzone_K"]
    for key, df in dfs.items():
        for var in variables_to_C:
            try:
                df[var[:-1] + "C"] = df[var] - 273.15
            except:
                pass
    return dfs


def compare_results(
    dfs,
    outputDir=r"C:\\Users\ina\Box Sync\Onderzoek\Projects\IBSPA Project 1\Destest development/190603_ComparisonStrobe/Results/",
    library="IDEAS",
    buildingTypology="SFD",
    buildingID="1",
    insulationStandard="1980s",
    occupant=None,
    selection=["1", "2", "3"],
):

    ### Select which results you want to plot, enter the values you want fixed
    dfsFocus = collections.OrderedDict()
    for key, value in dfs.items():
        libraryDF = key.split("_")[0]
        buildingTypologyDF = key.split("_")[1]
        buildingIDDF = key.split("_")[2]
        insulationStandardDF = key.split("_")[3]
        occupantDF = key.split("_")[4]

        if library is None:
            keyFocus = libraryDF
        elif buildingTypology is None:
            keyFocus = buildingTypologyDF
        elif buildingID is None:
            keyFocus = buildingIDDF
        elif insulationStandard is None:
            keyFocus = insulationStandardDF
        elif occupant is None:
            keyFocus = occupantDF
        else:
            keyFocus = None
            print("Please, make sure that the keyFocus variable has a value")
        if (
            (library is None or library == libraryDF)
            & (buildingTypology is None or buildingTypology == buildingTypologyDF)
            & (buildingID is None or buildingID == buildingIDDF)
            & (insulationStandard is None or insulationStandard == insulationStandardDF)
            & (occupant is None or occupant == occupantDF)
        ):
            dfsFocus.update(
                [(keyFocus, value)]
            )  # keyFocus is the distinguishing parameters, eg. if libraries compared, then it equals IDEAS or ...

    if selection is not None:
        # Not all variants should be included in the comparison
        dfsFocus2 = {k: dfsFocus[k] for k in selection}
        dfsFocus = dfsFocus2
        print("Only these items will be included in the comparison: " + str(selection))
    else:
        pass

    # Provide clear title for plot and the correct colors and linestyles for the plots
    if library is None:
        title = (
            "Results for different libraries \n ("
            + str(buildingTypology)
            + ", "
            + str(buildingID)
            + ", "
            + str(insulationStandard)
            + ", "
            + str(occupant)
            + ")"
        )
        # Provide colors and linestyles for the possible options that are there
        styles = {
            "IDEAS": ["darkgreen", "-"],
            "Buildings": ["tomato", "-"],
            "AixLib": ["navy", "-."],
            "BuildingSystems": ["cornflowerblue", "-."],
            "IDAICE": ["mediumseagreen", "--"],
            "DIMOSIM": ["darkred", "--"],
            "Trnsys": ["black", "-"],
            "Trnsys2": ["darkgrey", "-."],
        }
    elif buildingTypology is None:
        title = (
            "Results for different building typologies \n ("
            + str(library)
            + ", "
            + str(buildingID)
            + ", "
            + str(insulationStandard)
            + ", "
            + str(occupant)
            + ")"
        )
        styles = None
    elif buildingID is None:
        title = (
            "Results for different building IDs \n ("
            + str(library)
            + ", "
            + str(buildingTypology)
            + ", "
            + str(insulationStandard)
            + ", "
            + str(occupant)
            + ")"
        )
        styles = None
    elif insulationStandard is None:
        title = (
            "Results for different insulation standards \n ("
            + str(library)
            + ", "
            + str(buildingTypology)
            + ", "
            + str(buildingID)
            + ", "
            + str(occupant)
            + ")"
        )
        styles = {
            "1980s": ["tomato", "-"],
            "2000s": ["cornflowerblue", "-."],
            "2010s": ["mediumseagreen", "--"],
        }
    elif occupant is None:
        title = (
            "Results for different occupants \n ("
            + str(library)
            + ", "
            + str(buildingTypology)
            + ", "
            + str(buildingID)
            + ", "
            + str(insulationStandard)
            + ")"
        )
        styles = {
            "1": ["firebrick", "-"],
            "10": ["tomato", "-."],
            "11": ["firebrick", "--"],
            "12": ["tomato", ":"],
            "13": ["darkgreen", "-"],
            "14": ["mediumseagreen", "-."],
            "15": ["darkgreen", "--"],
            "16": ["mediumseagreen", ":"],
            "2": ["navy", "-"],
            "3": ["cornflowerblue", "-."],
            "4": ["navy", "--"],
            "5": ["cornflowerblue", ":"],
            "6": ["darkgrey", "-"],
            "7": ["lightgrey", "-."],
            "8": ["darkgrey", "--"],
            "9": ["lightgrey", ":"],
            "ISO": ["black", "-"],
        }
    else:
        title = None
        styles = None

    # Calc numbers
    calc_numbers(outputDir=outputDir, dfs=dfsFocus, styles=styles, title=title)

    # Plot profiles
    plot_profiles(outputDir=outputDir, dfs=dfsFocus, styles=styles, title=title)

    # Plot LDC
    plot_ldc(outputDir=outputDir, dfs=dfsFocus, styles=styles, title=title)


def plot_relative_difference_occupants(
    insulationStandard="2010s",
    occs=["1", "2", "3", "4", "5", "6", "9", "10", "11", "12", "13", "14", "15", "16"],
    libs=["AixLib", "Buildings", "DIMOSIM", "IDEAS"],
):
    params = {
        "legend.fontsize": "small",
        "figure.figsize": (15, 5),
        "axes.labelsize": "small",
        "axes.titlesize": "small",
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
    }
    plt.rcParams.update(params)

    styles = {
        "IDEAS": ["darkgreen", "-"],
        "Buildings": ["tomato", "-"],
        "AixLib": ["navy", "-."],
        "BuildingSystems": ["cornflowerblue", "-."],
        "IDAICE": ["mediumseagreen", "--"],
        "DIMOSIM": ["darkred", "--"],
        "Trnsys": ["black", "-"],
        "Trnsys2": ["darkgrey", "-."],
    }

    df_numbers = pd.read_csv(
        outputDir + "summary_allvariables.csv", sep=";", index_col=0
    )
    for lib in libs:
        for renov in [insulationStandard]:
            for (
                occ
            ) in (
                occs
            ):  # TODO: add number 8, it is not included in IDEAS, add number 7 as it is not included in AixLib
                df_numbers.loc[lib + "_SFD_1_" + renov + "_" + occ + "_rel", :] = (
                    df_numbers.loc[lib + "_SFD_1_" + renov + "_" + occ, :]
                    - df_numbers.loc[lib + "_SFD_1_" + renov + "_ISO", :]
                ) / df_numbers.loc[lib + "_SFD_1_" + renov + "_ISO", :]
    df_rel = df_numbers.loc[
        [ind for ind in df_numbers.index if insulationStandard in ind and "rel" in ind],
        :,
    ]

    fig, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(20, 12))
    for i, occ in enumerate(occs):
        keys = [
            lib + "_SFD_1_" + insulationStandard + "_" + occ + "_rel" for lib in libs
        ]
        print(occ)
        print(df_rel.loc[keys, "Energydemand_building_kWh"])
        df_rel.loc[keys, "Energydemand_building_kWh"].plot.bar(
            ax=axes.flat[i], color=[styles[i][0] for i in libs]
        )
        axes.flat[i].title.set_text(str(occ))
        axes.flat[i].set_ylabel("Percentage error \n annual energy demand")
    plt.setp(axes, xticklabels=libs)
    plt.suptitle(
        "Impact of different occupants for the "
        + insulationStandard
        + " buildings, relative to ISO occupant"
    )
    # plt.show()
    plt.savefig(outputDir + "Relative_occs_for" + insulationStandard + ".png", dpi=300)


if __name__ == "__main__":
    # TODO: pay attention! Debug function doesn't work properly for plotting, use run function
    # TODO: Also, pay attention with changing the df while looping > power is divided for LDC, but next time, this power is neglible, so fix or run one option at the time

    inputDir = r"C:\Users\u0110449\Box\Onderzoek\Projects\IBSPA Project 1\Destest development\simulation_results/"
    outputDirAll = r"C:\Users\u0110449\Box\Onderzoek\Projects\IBSPA Project 1\Destest development\200507/"
    if not os.path.exists(outputDirAll):
        os.makedirs(outputDirAll)

    ### GENERAL CODE
    # Using this script, you can create the plots you like. You have to choose what you'd like to compare.
    # Currently, you can only compare OR libraries OR typologies OR ids OR insulation standards OR occupants.
    # What you want to compare, needs to be None. The other parameters need to be fixed.
    # You can choose which of e.g. the libraries to compare by specifying them as a list in the "selection" variable.
    # E.g. not all libraries but only IDEAS and DIMOSIM or not all occupants but only 1, 2, 3
    ### First, import all results, then plot
    # dfs = import_results(inputDir)

    ### Compare different insulation standard for different occupants between libraries
    if True:
        insulationStandards = ["1980s", "2000s", "2010s"]
        occupants = [
            "ISO",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
        ]
        for insulationStandard in insulationStandards:
            for occupant in occupants:
                print(insulationStandard)
                print(occupant)
                # Make sure only 1 of library, buildingTypology, buildingID, insulationStandard, occupant
                library = (
                    None
                )  # Choose from: IDEAS, Buildings, AixLib, BuildingSystems, IDAICE, DIMOSIM, Trnsys
                buildingTypology = "SFD"  # Choose from: SFD
                buildingID = "1"  # Choose from: 1
                selection = (
                    None
                )  # ['1', '5', '9', '13']  # If you only want certain variants, add here
                compare_results(
                    dfs=dfs,
                    outputDir=outputDirAll + insulationStandard + "_" + occupant + "_",
                    library=library,
                    buildingTypology=buildingTypology,
                    buildingID=buildingID,
                    insulationStandard=insulationStandard,
                    occupant=occupant,
                    selection=selection,
                )

    # Compare relatives for different occupants
    if True:
        outputDir = outputDirAll + "ALL_"
        if not os.path.isfile(outputDir + "summary_allvariables.csv"):
            calc_numbers(dfs, outputDir=outputDir, styles=None, title=None)

        # Check influence of different occupants for the 1980s buildings compared to ISO occupant
        insulationStandard = "1980s"
        plot_relative_difference_occupants(insulationStandard=insulationStandard)
        insulationStandard = "2000s"
        plot_relative_difference_occupants(insulationStandard=insulationStandard)
        insulationStandard = "2010s"
        plot_relative_difference_occupants(insulationStandard=insulationStandard)

    print("IBSPA Project 1 WP3: That's it! :)")

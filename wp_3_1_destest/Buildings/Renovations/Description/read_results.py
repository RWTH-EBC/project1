"""Read results for IBPSA Project 1."""

import os
import pandas as pd
from dymola.dymola_interface import DymolaInterface


def results_to_csv(res_path, name):
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
    # import ipdb
    #
    # ipdb.set_trace()
    # results = results.drop(index_to_drop)
    # results = results.groupby(level=0).first()
    # results.to_csv(path=res_path, delimiter=';')
    dymola.close()

    # time = pd.to_numeric(results.index)
    # time -= 31536000
    # results.index = time
    # results = results.ix[0:31536000]

    res_csv = os.path.join(workspace, "{}.csv".format(name))

    results.to_csv(res_csv)
    print(results)
    print(res_csv)

    return results


if __name__ == "__main__":

    # workspace = os.path.join(
    #     "D:\\", "workspace", "results", "Simple_District_Destest_AixLib"
    # )
    # result_files = []
    # for f in os.listdir(workspace):
    #     if f.endswith(".mat"):
    #         # result_files.append(f)
    #         results_to_csv(os.path.join(workspace, f), name=f.replace(".mat", ""))
    RESULTS = [
        os.path.join(
            "D:\\",
            "workspace",
            "results",
            "Simple_District_Retrofit2000_Destest_AixLib",
        ),
        os.path.join(
            "D:\\",
            "workspace",
            "results",
            "Simple_District_Retrofit2000_Occ_Destest_AixLib",
        ),
        os.path.join(
            "D:\\",
            "workspace",
            "results",
            "Simple_District_Retrofit2010_Destest_AixLib",
        ),
        os.path.join(
            "D:\\",
            "workspace",
            "results",
            "Simple_District_Retrofit2010_Occ_Destest_AixLib",
        ),
    ]
    result_files = []
    for workspace in RESULTS:
        for f in os.listdir(workspace):
            if f.endswith(".mat"):
                # result_files.append(f)
                results_to_csv(os.path.join(workspace, f), name=f.replace(".mat", ""))

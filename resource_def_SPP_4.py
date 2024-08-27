# Order of operations:
# 1. Run the scripts that are on 'Documents/generator_gas_pricing'
# - run4.py (change the ISO_name to MISO or SWPP)
# - example_multi_outpout_regressor_2.py
# - present_results_.py)
# This will generate the following files:
# coefficients_NG_hub_name.pkl
# intercepts.pkl


# files that must be available:
# raw_training_data_with_target_{category}.csv
# generators_with_close_hubs.csv
# coefficients_NG_hub_name.pkl
# intercepts.pkl
# ML_modeling_results.pkl
# output.parquet

# what is new?
# Pmax using EIA data

# find all cases of miso by the following, and then order them to find the latest one:
# https://api1.marginalunit.com/reflow/spp-se/cases

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import NamedTuple
import pytz
from datetime import date
import io
import os

plt.rcParams["figure.figsize"] = (10, 7)
import requests
import duckdb
import json
import jsonplus
import pickle
import numpy as np

import economic_model_ML_funcs
from verify_mapping import get_new_mapping

jsonplus.prefer_compat()


# In[2]:


COLLECTION = "spp-se"
GO_TO_GCLOUD = True


def _get_auth(env_var: str = "SELF"):
    return tuple(os.environ[env_var].split(":"))


coefficients = pd.read_pickle("./coefficients_NG_hub_name.pkl")
intercepts = pd.read_pickle("./intercepts.pkl")

AUTH = _get_auth()


def _get_dfm(url, auth=AUTH):
    resp = requests.get(url, auth=auth)

    if resp.status_code != 200:
        print(resp.text)
        resp.raise_for_status()

    dfm = pd.read_csv(io.StringIO(resp.text))

    return dfm


# list_of_missing_plants = ['Cadiz Power Plant',
# 'Mattawoman Energy Center',
# "O'Brien Wind",
# 'Washington Parish Energy Center',
# 'Minco Wind V, LLC',
# 'Wildcat Wind Farm II LLC`',
# 'Ripley Westfield Wind LLC',
# 'Mason Dixon Wind Farm',
# 'Majestic 2 Wind Farm',
# 'Alpaca',
# 'Pioneer Crossing Energy, LLC',
# 'Maidencreek Biomass Plant',
# 'Lake Area Landfill Gas Recovery']

# list_of_missing_plants_with_start_dates  = ['Cadiz Power Plant', March 2016,
# 'Mattawoman Energy Center',    Not working
# "O'Brien Wind",
# 'Washington Parish Energy Center', November 2020,
# 'Minco Wind V, LLC', December 2018,
# 'Wildcat Wind Farm II LLC`',  Not working
# 'Ripley Westfield Wind LLC',    Cancelled
# 'Mason Dixon Wind Farm',
# 'Majestic 2 Wind Farm', August 2012,
# 'Alpaca', April 2017,
# 'Pioneer Crossing Energy, LLC', October 2008,
# 'Maidencreek Biomass Plant', Start: October 2008 - retired already but not clear when
# 'Lake Area Landfill Gas Recovery, February 2024']

# In[4]:

# in order to save cost of using the google cloud, we will download the data from the API and save it in a csv file
if GO_TO_GCLOUD:
    df_g_spp = _get_dfm(f"https://api1.marginalunit.com/reflow/{COLLECTION}/generators")
    df_g_spp.to_csv("spp-se-generator.csv", index=False)
else:
    df_g_spp = pd.read_csv("spp-se-generator.csv")


# In[5]:

# in order to save cost of using the google cloud, we will download the data from the API and save it in a csv file
if GO_TO_GCLOUD:
    df_p_spp = pd.read_gbq(
        """
        SELECT
        uid,
        APPROX_QUANTILES(pmin, 100)[OFFSET(50)] as pmin_median,
        APPROX_QUANTILES(pmax, 100)[OFFSET(50)] as pmax_median,
        APPROX_QUANTILES(pmax, 100)[OFFSET(99)] as pmax_99_perc,
        MAX(pmax) as pmax_max,
        FROM `vocal-door-162221.pr_forecast.spp-se-generator`
        GROUP BY uid
        ORDER BY pmax_median DESC
        """
    )
    # save the data in a csv file
    df_p_spp.to_csv("df_p_spp.csv", index=False)
else:
    df_p_spp = pd.read_csv("df_p_spp.csv")

# In[6]:

# in order to save cost of using the google cloud, we will download the data from the API and save it in a csv file
# if GO_TO_GCLOUD:
#     df_map_spp = _get_dfm(
#         f"https://api1.marginalunit.com/rms/eia/generators/{COLLECTION}/generator-mappings"
#     )
#     df_map_spp.to_csv("df_map_spp.csv", index=False)
# else:
#     df_map_spp = pd.read_csv("df_map_spp.csv")

df_map_spp = get_new_mapping(
    iso="SPP", use_mannual_override=True, use_additional_mapping=True
)

# if True:
#     # Adding Adrien's found generators, 2024-08-09
#     df_map_spp = df_map_spp.append({'generator_name': 'SIKESTON 161 12 SIKEGEN UN', 'eia_utility_id': 17177, 'eia_plant_code': 6768, 'eia_gen_id': 1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'SUB1363 161 TR1 TRTL1 UN', 'eia_utility_id': 14127, 'eia_plant_code': 64547, 'eia_gen_id': 1}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'SUB1363 161 TR2 TRTL2 UN', 'eia_utility_id': 14127, 'eia_plant_code': 64547, 'eia_gen_id': 2}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'ST_FRAN 16 2522 STFRANCIS02 UN', 'eia_utility_id': 924, 'eia_plant_code': 7604, 'eia_gen_id': 2}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'ST_FRAN 16 2521 STFRANCIS01 UN', 'eia_utility_id': 924, 'eia_plant_code': 7604, 'eia_gen_id': 1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'CBLUFFS 345 U42 WSEC_4_UNIT UN', 'eia_utility_id': 12341, 'eia_plant_code': 1082, 'eia_gen_id': 4}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'NENGLDBT 34.5 9001 NENGL_UNIT1 UN', 'eia_utility_id': 12341, 'eia_plant_code': 62133, 'eia_gen_id': 1}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'NENGLDBT 34.5 9001 NENGL_UNIT1 UN', 'eia_utility_id': 12341, 'eia_plant_code': 62133, 'eia_gen_id': 'GT1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'ARBHLDEZ 34.5 9001 ARBHL_UNIT1 UN', 'eia_utility_id': 12341, 'eia_plant_code': 62132, 'eia_gen_id': 1}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'ARBHLDEZ 34.5 9001 ARBHL_UNIT1 UN', 'eia_utility_id': 12341, 'eia_plant_code': 62132, 'eia_gen_id': 'WT1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'CHEMOL 115 7024 LSPOWE_1_UNIT UN', 'eia_utility_id': 55912, 'eia_plant_code': 55010, 'eia_gen_id': 'CTG1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'CHEMOL 115 7024 LSPOWE_1_UNIT UN', 'eia_utility_id': 55912, 'eia_plant_code': 55010, 'eia_gen_id': 'STG1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'ANSON 115 700Z EQ_UN_7002 UN', 'eia_utility_id': 13781, 'eia_plant_code': 7237, 'eia_gen_id': 1}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'ANSON 115 700Z EQ_UN_7002 UN', 'eia_utility_id': 13781, 'eia_plant_code': 7237, 'eia_gen_id': 2}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'ANSON 18 9100 ANSON_4_UNIT UN', 'eia_utility_id': 13781, 'eia_plant_code': 7237, 'eia_gen_id': 4}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'DOUD2 161 5022 GREATPATHFINDR UN', 'eia_utility_id': 64989, 'eia_plant_code': 65715, 'eia_gen_id': 'GPFWF'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'MANKATEC 15 9002 MANKAT_2_UNIT UN', 'eia_utility_id': 63705, 'eia_plant_code': 56104, 'eia_gen_id': 'CTG1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'MANKATEC 115 7007 MANKAT_1_UNIT UN', 'eia_utility_id': 63705, 'eia_plant_code': 56104, 'eia_gen_id': 'CTG2'}, ignore_index=True)
#
#     #remove a generator from df_map_spp that has the name 'MANKATEC 20 9008 MANKAT_3_UNIT2 UN' / MANKATEC 20 9007 MANKAT_3_UNIT1 UN
#     df_map_spp = df_map_spp[~df_map_spp.generator_name.str.contains('MANKATEC 20 9007 MANKAT_3_UNIT1 UN')]
#     df_map_spp = df_map_spp[~df_map_spp.generator_name.str.contains('MANKATEC 20 9008 MANKAT_3_UNIT2 UN')]
#
#     df_map_spp = df_map_spp.append({'generator_name': 'MANKATEC 20 9007 MANKAT_3_UNIT1 UN', 'eia_utility_id': 63705, 'eia_plant_code': 56104, 'eia_gen_id': 'STG1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'MANKATEC 20 9008 MANKAT_3_UNIT2 UN', 'eia_utility_id': 63705, 'eia_plant_code': 56104, 'eia_gen_id': 'STG1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'IDACWFIW 34.5 9009 IDACW_UNIT1 UN', 'eia_utility_id': 12341, 'eia_plant_code': 62795, 'eia_gen_id': 1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'FREEBORN 161 5000 WIND UN', 'eia_utility_id': 12341, 'eia_plant_code': 62795, 'eia_gen_id': 1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'STORY_CO 34.5 9008 STORYCO1_FPLP UN', 'eia_utility_id': 56119, 'eia_plant_code': 56924, 'eia_gen_id': 'GE15'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'INVRHLS 115 702Z EQ_UN_7020 UN', 'eia_utility_id': 13781, 'eia_plant_code': 1913, 'eia_gen_id': 1}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'INVRHLS 115 702Z EQ_UN_7020 UN', 'eia_utility_id': 13781, 'eia_plant_code': 1913, 'eia_gen_id': 2}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'INVRHLS 115 702Z EQ_UN_7020 UN', 'eia_utility_id': 13781, 'eia_plant_code': 1913, 'eia_gen_id': 3}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'ROCK_CO 34.5 9011 WALLEYEW UN', 'eia_utility_id': 65015, 'eia_plant_code': 65764, 'eia_gen_id': 'WWLLC'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'BUFFRID 34.5 9917 J545_UN_DIR UN', 'eia_utility_id': 65012, 'eia_plant_code': 65761, 'eia_gen_id': 3161}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'CLCG 69 8009 CLCG_2_UNIT UN', 'eia_utility_id': 772, 'eia_plant_code': 10860, 'eia_gen_id': '2A'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'CLCG 69 8008 CLCG_1_UNIT UN', 'eia_utility_id': 772, 'eia_plant_code': 10860, 'eia_gen_id': '1A'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'CARROLL 161 5027 CARROLL_2_UNIT UN', 'eia_utility_id': 12341, 'eia_plant_code': 56809, 'eia_gen_id': 'CWF'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'RICHLNDW 345 3004 SACCOWIND UN', 'eia_utility_id': 64261, 'eia_plant_code': 64662, 'eia_gen_id': 'SAC'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'WORT 34 9100 TOPIAW_UNIT UN', 'eia_utility_id': 15399, 'eia_plant_code': 56383, 'eia_gen_id': 'TOI2'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'BROOKE 69 934 STORMLK_2 UN', 'eia_utility_id': 59496, 'eia_plant_code': 55287, 'eia_gen_id': 'EXIS'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'KOSSUTHW 345 3100 J534_NEER UN', 'eia_utility_id': 64979, 'eia_plant_code': 65725, 'eia_gen_id': 'KCW'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'INPNDCEW 161 5003 INDEPENDENCEWF UN', 'eia_utility_id': 63802, 'eia_plant_code': 64183, 'eia_gen_id': 'IWE'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'PRESCOT 69 907 PRESCTW UN', 'eia_utility_id': 65857, 'eia_plant_code': 66952, 'eia_gen_id': 'PWE'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'DUANARN 34.5 3002 DA1_J504_SOL UN', 'eia_utility_id': 9417, 'eia_plant_code': 67140, 'eia_gen_id': 'PV1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'DUANARN 34.5 3003 DA2_R1020_SOL1 UN', 'eia_utility_id': 9417, 'eia_plant_code': 67537, 'eia_gen_id': 'PV1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'DUANARN 34.5 3007 DA2_R1020_SOL2 UN', 'eia_utility_id': 9417, 'eia_plant_code': 67537, 'eia_gen_id': 'PV1'}, ignore_index=True)
#     df_map_spp = df_map_spp.append({'generator_name': 'DUANARN 34.5 3008 DA2_R1020_SOL3 UN', 'eia_utility_id': 9417, 'eia_plant_code': 67537, 'eia_gen_id': 'PV1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'CRESTNR 69 921 CRESESOL UN', 'eia_utility_id': 9417, 'eia_plant_code': 67536, 'eia_gen_id': 'PV1'}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'OAKGLNW 34.5 9006 G604_UNIT UN', 'eia_utility_id': 12667, 'eia_plant_code': 57432, 'eia_gen_id':1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'JEFFWDNS 69 8002 JEFFWD_2_UNIT UN', 'eia_utility_id': 13781, 'eia_plant_code': 56750, 'eia_gen_id':1}, ignore_index=True)
#
#     df_map_spp = df_map_spp.append({'generator_name': 'BOND_SS 34.5 9001 H007_UNIT UN', 'eia_utility_id': 60025, 'eia_plant_code': 57417, 'eia_gen_id':'NORD'}, ignore_index=True)


# In[8]:


# df_hc = pd.read_csv("/Users/michael.simantov/Downloads/unit_heat_input_curve_2024-04-11-MISO.csv")


# In[ ]:


# TO BE UPDATED FOR MISO
with open("./resources_SPP_epa.json", "r") as fdesc:
    epa_eia_mapping = json.load(fdesc)

records = [
    {
        "orispl": r["orispl"],
        "epa_generator_id": r["generator_id"],
        "eia_unit_id": u["eia_unit"]["generator_id"],
    }
    for r in epa_eia_mapping
    for u in r["weighted_eia_units"]
]

df_eia_epa = pd.DataFrame(records)


# In[ ]:


# in order to save cost of using the google cloud, we will download the data from the API and save it in a csv file
if GO_TO_GCLOUD:
    df_eia_spp = _get_dfm(
        "https://api1.marginalunit.com/misc-data/eia/generators/monthly?columns=plant_id,plant_name,plant_state,generator_id,energy_source_code,prime_mover_code,operating_month,operating_year,latitude,longitude,retirement_month,retirement_year,planned_operation_month,planned_operation_year,net_summer_capacity_mw"
    )
    df_eia_spp.to_csv("df_eia_spp.csv", index=False)
else:
    df_eia_spp = pd.read_csv("df_eia_spp.csv")


# In[ ]:


dfm = pd.merge(
    df_g_spp.set_index("uid", verify_integrity=True),
    df_p_spp.set_index("uid", verify_integrity=True),
    left_index=True,
    right_index=True,
).sort_values("pmax_median", ascending=False)


# In[ ]:


# merge only the column 'net_summer_capacity_mw' from df_eia_spp to df_map_spp based on df_map_spp['eia_plant_code']== and df_eia_spp['plant_id'] and df_map_spp['eia_gen_id']== and df_eia_spp['generator_id']
columns_to_keep = df_map_spp.columns.tolist() + ["net_summer_capacity_mw"]
df_map_spp = pd.merge(
    df_map_spp,
    df_eia_spp,
    left_on=["eia_plant_code", "eia_gen_id"],
    right_on=["plant_id", "generator_id"],
    how="left",
)
df_map_spp = df_map_spp[columns_to_keep]

# add to df_map_spp a column that is named 'agg' and its value is the sum of all the net_summer_capacity_mw values that have the same generator_name
df_map_spp["agg"] = df_map_spp.groupby("generator_name")[
    "net_summer_capacity_mw"
].transform("sum")
df_map_spp.drop(columns=["net_summer_capacity_mw"], inplace=True)
df_map_spp.rename(columns={"agg": "net_summer_capacity_mw"}, inplace=True)


dfm = pd.merge(
    dfm,
    df_map_spp.groupby("generator_name").first(),
    left_index=True,
    right_index=True,
    how="left",
).sort_values("pmax_median", ascending=False)


# In[ ]:


dfm = pd.merge(
    dfm,
    df_eia_epa.groupby(["orispl", "eia_unit_id"]).first(),
    left_on=["eia_plant_code", "eia_gen_id"],
    right_index=True,
    how="left",
)
dfm.epa_generator_id = dfm.epa_generator_id.fillna(dfm.eia_gen_id)


# In[ ]:


# dfm = pd.merge(
#     dfm,
#     df_hc.set_index(["orispl", "unit"], verify_integrity=True)[["fitting_param_x", "fitting_param_intercept"]],
#     left_on=["eia_plant_code", "epa_generator_id"],
#     right_index=True,
#     how="left"
# )
# dfm["fitting_param_x"] = 10
# dfm["fitting_param_intercept"] = 100
dfm["fitting_param_x"] = 0.0
dfm["fitting_param_intercept"] = 0.0


# In[ ]:


dfm = pd.merge(
    dfm,
    df_eia_spp[~df_eia_spp.plant_id.isnull()].set_index(
        ["plant_id", "generator_id"], verify_integrity=True
    ),
    left_on=["eia_plant_code", "eia_gen_id"],
    right_index=True,
    how="left",
)


dfm.drop(columns=["net_summer_capacity_mw_y"], inplace=True)
dfm.rename(columns={"net_summer_capacity_mw_x": "net_summer_capacity_mw"}, inplace=True)

# In[ ]:


# 1) Remove generators that we have not seen in more than 1 year
dfm = dfm[dfm.latest_appearance >= "stateestimator_20230101"]

if False:
    # plotting dfm.pmax_99_perc vs. dfm.net_summer_capacity_mw where net_summer_capacity_mw is not null
    plt.scatter(dfm.net_summer_capacity_mw, dfm.pmax_99_perc)
    plt.xlabel("net_summer_capacity_mw", fontsize=24)
    plt.ylabel("pmax_99_perc", fontsize=24)
    plt.axis([0, 2500, 0, 2500])
    plt.figure()
    # creating a histogram of dfm.net_summer_capacity_mw - dfm.pmax_99_perc. The count should be in percent
    plt.hist(dfm.net_summer_capacity_mw - dfm.pmax_99_perc, bins=100)
    plt.xlabel("(net_summer_capacity_mw) - (pmax_99_perc)", fontsize=24)
    plt.ylabel("count", fontsize=24)
    plt.show()


# replace dfm.pmax_99_perc with dfm.net_summer_capacity_mw where net_summer_capacity_mw is not null and pmax_99_perc is not null and net_summer_capacity_mw < dfm.pmax_99_perc
dfm.loc[
    ~dfm.net_summer_capacity_mw.isnull()
    & ~dfm.pmax_99_perc.isnull()
    & (dfm.net_summer_capacity_mw < dfm.pmax_99_perc)
    & (dfm.net_summer_capacity_mw > dfm.pmin_median),
    "pmax_99_perc",
] = dfm.net_summer_capacity_mw

# replace dfm.pmax_99_perc with dfm.net_summer_capacity_mw where net_summer_capacity_mw is not null and pmax_99_perc is null
dfm.loc[
    ~dfm.net_summer_capacity_mw.isnull() & dfm.pmax_99_perc.isnull(), "pmax_99_perc"
] = dfm.net_summer_capacity_mw

dfm.drop(columns=["net_summer_capacity_mw"], inplace=True)


# In[ ]:

# # Manually adding missing data for plants
# dfm.loc[dfm.plant_name == "Cadiz Power Plant", "operating_month"] = 3
# dfm.loc[dfm.plant_name == "Cadiz Power Plant", "operating_year"] = 2016
# dfm.loc[dfm.plant_name == "Washington Parish Energy Center", "operating_month"] = 11
# dfm.loc[dfm.plant_name == "Washington Parish Energy Center", "operating_year"] = 2020
# dfm.loc[dfm.plant_name == "Minco Wind V, LLC", "operating_month"] = 12
# dfm.loc[dfm.plant_name == "Minco Wind V, LLC", "operating_year"] = 2018
# dfm.loc[dfm.plant_name == "Majestic 2 Wind Farm", "operating_month"] = 8
# dfm.loc[dfm.plant_name == "Majestic 2 Wind Farm", "operating_year"] = 2012
# dfm.loc[dfm.plant_name == "Alpaca", "operating_month"] = 4
# dfm.loc[dfm.plant_name == "Alpaca", "operating_year"] = 2017
# dfm.loc[dfm.plant_name == "Pioneer Crossing Energy, LLC", "operating_month"] = 10
# dfm.loc[dfm.plant_name == "Pioneer Crossing Energy, LLC", "operating_year"] = 2008
# dfm.loc[dfm.plant_name == "Lake Area Landfill Gas Recovery", "operating_month"] = 2
# dfm.loc[dfm.plant_name == "Lake Area Landfill Gas Recovery", "operating_year"] = 2024

dfm_original__ = dfm.copy()


# print plant names that are already missing from the list of known missing plants
# The following plants were dropped because they do not have data about their operating year and month
# for item in list_of_missing_plants:
#     if not any(dfm.plant_name.str.contains(item)):
#         print(item)


# dfm_to_map = dfm[dfm.eia_plant_code.isnull()]
dfm.to_csv("df_spp_sub_mapping.csv", index=False)
dfm.reset_index().to_csv("spp_plant_name_and_coordinates.csv", index=False)

dfm_1 = dfm[~dfm.energy_source_code.isnull()].copy()


# In[ ]:


avg_heat_rate = dfm_1.groupby(["energy_source_code", "prime_mover_code"]).agg(
    {
        "fitting_param_x": "median",
        "fitting_param_intercept": "median",
        "pmax_max": "max",
    }
)


# In[ ]:


dfm_1.loc[:, ["fitting_param_x", "fitting_param_intercept"]] = dfm_1.apply(
    lambda r: (
        avg_heat_rate.loc[(r.energy_source_code, r.prime_mover_code)]
        if pd.isnull(r.fitting_param_x)
        else r[["fitting_param_x", "fitting_param_intercept"]]
    ),
    axis=1,
)


# In[ ]:


ignore_missing_heat_rates = {"SUN", "WAT", "WND", "MWH", "NUC"}

default_hr = {("NG", "IC"): (8.9, 120)}

for key, value in default_hr.items():
    energy_source, prime_mover = key
    fitting_param_x, fitting_param_intercept = value

    dfm_1.loc[
        (dfm_1.energy_source_code == energy_source)
        & (dfm_1.prime_mover_code == prime_mover),
        ["fitting_param_x", "fitting_param_intercept"],
    ].fillna(
        {
            "fitting_param_x": fitting_param_x,
            "fitting_param_intercept": fitting_param_intercept,
        },
        inplace=True,
    )

# TODO: change that

# dfm_1 = dfm_1.fillna({"fitting_param_x": 10.0, "fitting_param_intercept": 100.0})
dfm_1 = dfm_1.fillna({"fitting_param_x": 0.0, "fitting_param_intercept": 0.0})

# In[ ]:

# Clean up
dfm_2 = dfm_1[
    dfm_1.energy_source_code.isin(ignore_missing_heat_rates)
    | ~dfm_1.fitting_param_x.isnull()
].copy()


# print plant names that are already missing from the list of known missing plants
# The following plants were dropped because they do not have data about their operating year and month
# for item in list_of_missing_plants:
#     if not any(dfm_2.plant_name.str.contains(item)):
#         print(item)

# dfm_2 = dfm_2[~dfm_2.operating_month.isnull()].copy()
# in ERCOT it is replaced by this line:
dfm_2 = dfm_2[
    ~dfm_2.operating_month.isnull() | ~dfm_2.planned_operation_month.isnull()
].copy()


# print plant names that are already missing from the list of known missing plants
# The following plants were dropped because they do not have data about their operating year and month
# for item in list_of_missing_plants:
#     if not any(dfm_2.plant_name.str.contains(item)):
#         print(item)


# In[ ]:


# Start up cost
# We should leverage Jieyu's model, especially for other iso
# for now, i'll just get the current start up cost

# !gsutil cp "gs://marginalunit-placebo-metadata/metadata/ercot.resourcedb/2023-08-16/resourcedb.zip" . && unzip resourcedb.zip
# df_rdb = pd.read_parquet("resources.parquet")
# df_gdb = pd.read_parquet("generators.parquet")


# In[ ]:


_NO_LIMIT = (1, 1, 9999)

# This is a default, we should do a proper research on this
_DEFAULT_PHYSICAL_CHARACTERISTICS = {
    # Renewable
    ("WND", "WT"): _NO_LIMIT,
    ("SUN", "PV"): _NO_LIMIT,
    # Batteries
    ("MWH", "BA"): _NO_LIMIT,
    # Hydro
    ("WAT", "HY"): _NO_LIMIT,
    # Natural Gas
    ("NG", "CT"): (4, 2, 200),
    ("NG", "GT"): (4, 2, 200),
    ("NG", "CS"): (4, 2, 200),
    ("NG", "ST"): (4, 2, 200),
    ("NG", "CA"): (4, 2, 200),
    ("NG", "IC"): (4, 2, 200),
    # Coal
    ("SUB", "ST"): (24, 24, 200),
    ("LIG", "ST"): (24, 24, 200),
    # Nuclear
    ("NUC", "ST"): (24, 24, 200),
    # Oil
    ("DFO", "GT"): (24, 24, 200),
    # Wood waste
    ("WDS", "ST"): _NO_LIMIT,
}

# This is a default, we should do a proper research on this, and use a proper model
_DEFAULT_START_UP_COST = {
    # Renewable
    ("WND", "WT"): 0.0,
    ("SUN", "PV"): 0.0,
    # Batteries
    ("MWH", "BA"): 0.0,
    # Hydro
    ("WAT", "HY"): 0.0,
    # Natural Gas
    ("NG", "CT"): 5_000,
    ("NG", "GT"): 5_000,
    ("NG", "CS"): 5_000,
    ("NG", "ST"): 5_000,
    ("NG", "CA"): 5_000,
    ("NG", "IC"): 5_000,
    # Coal
    ("SUB", "ST"): 2_000,
    ("LIG", "ST"): 2_000,
    # Nuclear
    ("NUC", "ST"): 0.0,
    # Oil
    ("DFO", "GT"): 500,
    # Wood waste
    ("WDS", "ST"): 0.0,
}

# no COAL or NG
_DEFAULT_PRICE = {
    # NG
    ("NG", "CT"): 18.0,
    ("NG", "GT"): 18.0,
    ("NG", "CS"): 18.0,
    ("NG", "ST"): 18.0,
    ("NG", "CA"): 18.0,
    ("NG", "IC"): 18.0,
    ("NG", "FC"): 18.0,
    # Coal
    ("SUB", "ST"): 20.0,
    ("LIG", "ST"): 20.0,
    ("AB", "ST"): 97.0,
    ("BIT", "ST"): 31.0,
    ("LIG", "ST"): 12.0,
    ("PC", "ST"): 47.6,
    ("RC", "ST"): 13.0,
    ("SUB", "ST"): 19.3,
    ("WC", "ST"): 13.3,
    # Renewable
    ("WND", "WT"): -30.0,
    ("SUN", "PV"): 0.0,
    # Batteries
    ("MWH", "BA"): 0.0,
    # Hydro
    ("WAT", "HY"): 0.0,
    # Oil
    ("DFO", "GT"): 16.5,  # was 20, but changed as per the data from Scott Bruns
    ("RFO", "ST"): 31.5,  # added based on data from Scott Bruns
    ("DFO", "ST"): 16.5,  # added based on data from Scott Bruns
    ("DFO", "CT"): 16.5,  # added based on data from Scott Bruns
    ("DFO", "CA"): 16.5,  # added based on data from Scott Bruns
    ("KER", "GT"): 20.0,  # added based on data from Scott Bruns
    ("OG", "OT"): 55.0,  # added based on data from Scott Bruns
    ("OG", "CA"): 55.0,  # added based on data from Scott Bruns
    ("OG", "ST"): 55.0,  # added based on data from Scott Bruns
    ("DFO", "IC"): 16.5,  # added based on data from Scott Bruns
    ("JF", "GT"): 20.0,  # added based on data from Scott Bruns
    # Nuclear
    ("NUC", "ST"): 0.0,
    # Wood waste
    ("WDS", "ST"): 0.0,
    # Other
    ("PUR", "ST"): 20.0,
    ("BLQ", "ST"): 45.0,
    ("WH", "ST"): 3.5,  # added based on data from Scott Bruns
    ("WH", "OT"): 3.5,  # added based on data from Scott Bruns
    ("LFG", "ST"): 15.0,  # added based on data from Scott Bruns
    ("LFG", "IC"): 15.0,  # added based on data from Scott Bruns
    ("LFG", "CT"): 15.0,  # added based on data from Scott Bruns
    ("LFG", "GT"): 15.0,  # added based on data from Scott Bruns
    ("OBG", "IC"): 15.0,  # added based on data from Scott Bruns
}


# In[ ]:


resources = []
supply_curves = {}

# find how many unique entries there are in dfm_2 for each combination of ('energy_source_code', 'prime_mover_code'):
dfm_2.groupby(["energy_source_code", "prime_mover_code"]).size()

# Michael adds this, to find relations between (energy_source, prime_mover) and physical properties

# make a copy of dfm_2 that we will use to create the model
dfm_2_copy = dfm_2.copy()

# Expanded category mapping for energy source codes
category_mapping = {
    "AB": "coal",
    "BFG": "biogas",
    "BIT": "coal",
    "BLQ": "biomass",
    "DFO": "oil",
    "JF": "oil",
    "KER": "oil",
    "LFG": "biogas",
    "LIG": "coal",
    "MSW": "waste",
    "MWH": "hydro",
    "NG": "gas",
    "NUC": "nuclear",
    "OBG": "biogas",
    "OG": "oil",
    "PC": "coal",
    "PUR": "other",
    "RC": "coal",
    "RFO": "oil",
    "SGC": "gas",
    "SUB": "coal",
    "SUN": "solar",
    "WAT": "hydro",
    "WC": "coal",
    "WDS": "biomass",
    "WH": "waste heat",
    "WND": "wind",
}

# Sample lists
list1 = [
    ("AB", "ST"),
    ("BFG", "ST"),
    ("BIT", "ST"),
    ("BLQ", "ST"),
    ("DFO", "CA"),
    ("DFO", "CT"),
    ("DFO", "GT"),
    ("DFO", "IC"),
    ("DFO", "ST"),
    ("JF", "GT"),
    ("KER", "GT"),
    ("LFG", "CT"),
    ("LFG", "GT"),
    ("LFG", "IC"),
    ("LFG", "ST"),
    ("LIG", "ST"),
    ("MSW", "ST"),
    ("MWH", "BA"),
    ("MWH", "FW"),
    ("NG", "CA"),
    ("NG", "CS"),
    ("NG", "CT"),
    ("NG", "FC"),
    ("NG", "GT"),
    ("NG", "IC"),
    ("NG", "ST"),
    ("NUC", "ST"),
    ("OBG", "IC"),
    ("OG", "CA"),
    ("OG", "OT"),
    ("OG", "ST"),
    ("PC", "ST"),
    ("PUR", "ST"),
    ("RC", "ST"),
    ("RFO", "ST"),
    ("SGC", "CA"),
    ("SGC", "CT"),
    ("SUB", "ST"),
    ("SUN", "PV"),
    ("WAT", "HY"),
    ("WAT", "PS"),
    ("WC", "ST"),
    ("WDS", "ST"),
    ("WH", "OT"),
    ("WH", "ST"),
    ("WND", "WS"),
    ("WND", "WT"),
]
list2 = [
    ("DFO", "GT"),
    ("LIG", "ST"),
    ("MWH", "BA"),
    ("NG", "CA"),
    ("NG", "CS"),
    ("NG", "CT"),
    ("NG", "GT"),
    ("NG", "ST"),
    ("NUC", "ST"),
    ("SUB", "ST"),
    ("SUN", "PV"),
    ("WAT", "HY"),
    ("WDS", "ST"),
    ("WND", "WT"),
]


# Function to get the category of an energy source code
def get_category(code):
    return category_mapping.get(code, "other")


# Function to find the closest pair
def find_closest_pair(pair, candidates):
    source_code, prime_mover = pair
    source_category = get_category(source_code)

    # Find the closest match by category
    closest_pair = None
    for candidate in candidates:
        candidate_code, candidate_mover = candidate
        if get_category(candidate_code) == source_category:
            closest_pair = candidate
            break
    return closest_pair


# Create the result list
results = {}

# Iterate through the first list and find matches or closest matches
for pair in list1:
    source_code, prime_mover = pair
    source_category = get_category(source_code)
    results[pair] = source_category

# Create the result list for two categories
results_2_lists = {}

# Iterate through the first and second list and find matches or closest matches for a pair
for pair in list2:
    closest_pair = find_closest_pair(pair, list1)
    results_2_lists[pair] = results[closest_pair]


####################################################### P_MIN
# Calculation of P_MIN

# add a column to dfm_2 with the category, based on the results:
# dfm_2['category'] = dfm_2.apply(lambda row: results[(row.energy_source_code, row.prime_mover_code)], axis=1)
dfm_2["category"] = dfm_2.apply(
    lambda row: results.get((row.energy_source_code, row.prime_mover_code), "other"),
    axis=1,
)

# convert the value in the column 'operating_year' to 2024 - the value in the column 'operating_year'
dfm_2["operating_year"] = 2024 - dfm_2.operating_year

# read the file that contains the ML_modeling_results that was created using ERCOT data
with open("ML_modeling_results.pkl", "rb") as fdesc:
    ML_modeling_results = pickle.load(fdesc)

# eliminate the column "operating month", "latitude" and "longitude", eia_utility_id and eia_plant_code from the dataframe dfm_2
dfm_2 = dfm_2.drop(
    columns=[
        "operating_month",
        "latitude",
        "longitude",
        "eia_utility_id",
        "eia_plant_code",
        "plant_state",
        "plant_name",
    ]
)


# use the functions in economic_model_ML_funcs to predict pmin_median. This should be done for each category that appears more than 20 times in the dataframe
for category in ML_modeling_results.keys():
    raw_testing_data = dfm_2[dfm_2.category == category].drop(columns=["pmin_median"])
    raw_training_data_with_target = pd.read_csv(
        f"raw_training_data_with_target_{category}.csv"
    )

    # find columns in raw_testing_data that are not in raw_training_data_with_target
    # missing_columns = list(set(raw_testing_data.columns) - set(raw_training_data_with_target.columns))

    # Preprocess raw data, to impute NaN, scale numerical features and remove target bias
    (
        processed_training_data,
        processed_testing_data,
        target,
        target_bias,
        target_std,
        numeric_indices,
        all_titles_of_features,
    ) = economic_model_ML_funcs.preprocess(
        None,
        None,
        "pmin_median",
        raw_training_data_with_target,
        raw_testing_data,
        False,
    )
    if processed_training_data is None:
        continue

    # create models on reduced train datasets
    (
        selected_train_data_linear,
        model_linear,
        scores_reduced_linear,
        selected_test_data_linear,
        chosen_indices_linear,
    ) = economic_model_ML_funcs.build_model_on_reduced_data(
        "linear", processed_training_data, target, processed_testing_data
    )
    (
        selected_train_data_XGB,
        model_XGB,
        scores_reduced_XGB,
        selected_test_data_XGB,
        chosen_indices_XGB,
    ) = economic_model_ML_funcs.build_model_on_reduced_data(
        "XGB", processed_training_data, target, processed_testing_data
    )

    # keep the ML modeling results, to be used on different markets:
    # ML_modeling_results[category] = {'model_XGB': model_XGB, 'model_linear': model_linear, 'target_bias': target_bias, 'chosen_indices_XGB': chosen_indices_XGB, 'chosen_indices_linear': chosen_indices_linear, 'target_std': target_std, 'numeric_indices': numeric_indices, 'all_titles_of_features': all_titles_of_features}

    Chosen_features_XGBoost = all_titles_of_features[chosen_indices_XGB]
    Chosen_features_linear = all_titles_of_features[chosen_indices_linear]

    # Check overfitting of models that were created on reduced data
    result_XGB, result_linear = economic_model_ML_funcs.check_model_performance(
        selected_train_data_XGB,
        selected_train_data_linear,
        scores_reduced_XGB,
        scores_reduced_linear,
        model_XGB,
        model_linear,
        target,
    )
    print(result_XGB)
    print(result_linear)

    economic_model_ML_funcs.report_R_squared(
        target, scores_reduced_linear, scores_reduced_XGB
    )

    prediction_linear = economic_model_ML_funcs.make_predictions(
        "linear",
        target_bias,
        target_std,
        selected_test_data_linear,
        selected_test_data_XGB,
        model_XGB,
        model_linear,
    )
    prediction_XGB = economic_model_ML_funcs.make_predictions(
        "XGB",
        target_bias,
        target_std,
        selected_test_data_linear,
        selected_test_data_XGB,
        model_XGB,
        model_linear,
    )

    # add the predicted values to the dataframe dfm_2
    dfm_2.loc[dfm_2.category == category, "pmin_median"] = prediction_XGB

    # print the importance of each feature in the model
    print(
        pd.Series(
            model_XGB.feature_importances_, index=Chosen_features_XGBoost
        ).sort_values(ascending=False)
    )

dfm_2_copy["pmin_median"] = dfm_2["pmin_median"]

# Coal
# # in rows where the category is 'coal', wherever the pmin_median is more than 55% of the pmax_99_perc, set the pmin_median to 55% of the pmax_99_perc
# dfm_2_copy.loc[(dfm_2.category == 'coal') & (dfm_2.pmin_median > 0.55 * dfm_2.pmax_99_perc), 'pmin_median'] = 0.55 * dfm_2.pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "coal") & (dfm_2.pmin_median > 0.8 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.8 * dfm_2.pmax_99_perc
)

# # in rows where the category is 'coal', wherever the pmin_median is less than 20% of the pmax_99_perc, set the pmin_median to 20% of the pmax_99_perc
# dfm_2_copy.loc[(dfm_2.category == 'coal') & (dfm_2.pmin_median < 0.20 * dfm_2.pmax_99_perc), 'pmin_median'] = 0.20 * dfm_2.pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "coal")
    & (dfm_2.prime_mover_code == "ST")
    & (dfm_2.pmin_median < 0.40 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.40 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "coal") & (dfm_2.pmin_median < 0.40 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.40 * dfm_2.pmax_99_perc
)  # for all other prime_mover_codes

# Gas
# # in rows where the category is 'gas', wherever the pmin_median is more than 55% of the pmax_99_perc, set the pmin_median to 55% of the pmax_99_perc
# dfm_2_copy.loc[(dfm_2.category == 'gas') & (dfm_2.pmin_median > 0.65 * dfm_2.pmax_99_perc), 'pmin_median'] = 0.65 * dfm_2.pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "gas") & (dfm_2.pmin_median > 0.65 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.65 * dfm_2.pmax_99_perc
)

# # in rows where the category is 'gas', wherever the pmin_median is less than 20% of the pmax_99_perc, set the pmin_median to 20% of the pmax_99_perc
# dfm_2_copy.loc[(dfm_2.category == 'gas') & (dfm_2.pmin_median < 0.20 * dfm_2.pmax_99_perc), 'pmin_median'] = 0.20 * dfm_2.pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "gas")
    & (dfm_2.prime_mover_code == "ST")
    & (dfm_2.pmin_median < 0.35 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.35 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "gas")
    & (dfm_2.prime_mover_code == "CT")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "gas")
    & (dfm_2.prime_mover_code == "CA")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "gas")
    & (dfm_2.prime_mover_code == "GT")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "gas") & (dfm_2.pmin_median < 0.35 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.35 * dfm_2.pmax_99_perc
)  # for all other prime_mover_codes

# Oil
# # in rows where the category is 'oil', wherever the pmin_median is more than 55% of the pmax_99_perc, set the pmin_median to 55% of the pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "oil") & (dfm_2.pmin_median > 0.85 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.85 * dfm_2.pmax_99_perc
)

# # in rows where the category is 'oil', wherever the pmin_median is less than 20% of the pmax_99_perc, set the pmin_median to 20% of the pmax_99_perc
# dfm_2_copy.loc[(dfm_2.category == 'oil') & (dfm_2.pmin_median < 0.50 * dfm_2.pmax_99_perc), 'pmin_median'] = 0.50 * dfm_2.pmax_99_perc
dfm_2_copy.loc[
    (dfm_2.category == "oil")
    & (dfm_2.prime_mover_code == "ST")
    & (dfm_2.pmin_median < 0.35 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.35 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "oil")
    & (dfm_2.prime_mover_code == "CA")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "oil")
    & (dfm_2.prime_mover_code == "CT")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "oil")
    & (dfm_2.prime_mover_code == "GT")
    & (dfm_2.pmin_median < 0.5 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.5 * dfm_2.pmax_99_perc
)
dfm_2_copy.loc[
    (dfm_2.category == "oil") & (dfm_2.pmin_median < 0.35 * dfm_2.pmax_99_perc),
    "pmin_median",
] = (
    0.35 * dfm_2.pmax_99_perc
)  # for all other prime_mover_codes

# renewbles and other
dfm_2_copy.loc[dfm_2.category == "hydro", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "hydro", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "nuclear", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "nuclear", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "wind", "pmin_median"] = 0
dfm_2_copy.loc[dfm_2.category == "biogas", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "biogas", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "biomass", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "biomass", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "waste", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "waste", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "other", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "other", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "solar", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "solar", "pmin_median"
].clip(lower=0)
dfm_2_copy.loc[dfm_2.category == "waste heat", "pmin_median"] = dfm_2_copy.loc[
    dfm_2.category == "waste heat", "pmin_median"
].clip(lower=0)


# set dfm_2_copy's value where the category is 'nuclear' to the maximum between the current value and 0:


# transfer the values of dfm_2.pmin_median to dfm_2_copy


dfm_2 = dfm_2_copy

# dfm_2.groupby(['energy_source_code', 'prime_mover_code'])[['pmin_median', 'pmax_99_perc', 'pmax_max']].agg(['median', 'max', 'min', 'count'])

####################################################### P_MIN       (end)
####################################################### INTERCEPT (for HEAT RATE)   (start)
# Calculation of HEAT RATE


# make a copy of dfm_2 that we will use to create the model
dfm_2_copy = dfm_2.copy()

# add a column to dfm_2 with the category, based on the results:
# dfm_2['category'] = dfm_2.apply(lambda row: results[(row.energy_source_code, row.prime_mover_code)], axis=1)
dfm_2["category"] = dfm_2.apply(
    lambda row: results.get((row.energy_source_code, row.prime_mover_code), "other"),
    axis=1,
)

# convert the value in the column 'operating_year' to 2024 - the value in the column 'operating_year'
dfm_2["operating_year"] = 2024 - dfm_2.operating_year

# read the file that contains the ML_modeling_results that was created using ERCOT data
with open("ML_modeling_results.pkl", "rb") as fdesc:
    ML_modeling_results = pickle.load(fdesc)

# eliminate the column "operating month", "latitude" and "longitude", eia_utility_id and eia_plant_code from the dataframe dfm_2
dfm_2 = dfm_2.drop(
    columns=[
        "operating_month",
        "latitude",
        "longitude",
        "eia_utility_id",
        "eia_plant_code",
        "plant_state",
        "plant_name",
    ]
)


# use the functions in economic_model_ML_funcs to predict pmin_median. This should be done for each category that appears more than 20 times in the dataframe
for category in ML_modeling_results.keys():
    raw_testing_data_ALL = dfm_2[dfm_2.category == category].drop(
        columns=["fitting_param_intercept"]
    )  # fitting_param_x
    raw_training_data_with_target_ALL = pd.read_csv(
        f"raw_training_data_with_target_{category}.csv"
    )

    # for each unique pair of 'energy_source_code' and 'prime_mover_code', find the number of rows in raw_testing_data_ALL that have that pair
    counts = raw_training_data_with_target_ALL.groupby(
        ["energy_source_code", "prime_mover_code"]
    ).size()

    # for each unique pair of 'energy_source_code' and 'prime_mover_code', if the number of rows in raw_testing_data_ALL if less than 20 then continue
    for pair in counts.index:
        print("number of points to work on: ", counts[pair])
        if counts[pair] < 20:
            continue

        # we want to work with each pair separately. Keep only the rows in raw_testing_data_ALL that have the pair
        raw_testing_data = raw_testing_data_ALL[
            (raw_testing_data_ALL.energy_source_code == pair[0])
            & (raw_testing_data_ALL.prime_mover_code == pair[1])
        ]
        raw_training_data_with_target = raw_training_data_with_target_ALL[
            (raw_training_data_with_target_ALL.energy_source_code == pair[0])
            & (raw_training_data_with_target_ALL.prime_mover_code == pair[1])
        ]

        # if the number of rows in raw_testing_data is 0, then continue
        if raw_testing_data.shape[0] == 0:
            continue

        # find columns in raw_testing_data that are not in raw_training_data_with_target
        # missing_columns = list(set(raw_testing_data.columns) - set(raw_training_data_with_target.columns))

        # Preprocess raw data, to impute NaN, scale numerical features and remove target bias
        (
            processed_training_data,
            processed_testing_data,
            target,
            target_bias,
            target_std,
            numeric_indices,
            all_titles_of_features,
        ) = economic_model_ML_funcs.preprocess(
            None,
            None,
            "fitting_param_intercept",
            raw_training_data_with_target,
            raw_testing_data,
            False,
        )
        if processed_training_data is None:
            continue

        # create models on reduced train datasets
        (
            selected_train_data_linear,
            model_linear,
            scores_reduced_linear,
            selected_test_data_linear,
            chosen_indices_linear,
        ) = economic_model_ML_funcs.build_model_on_reduced_data(
            "linear", processed_training_data, target, processed_testing_data
        )
        (
            selected_train_data_XGB,
            model_XGB,
            scores_reduced_XGB,
            selected_test_data_XGB,
            chosen_indices_XGB,
        ) = economic_model_ML_funcs.build_model_on_reduced_data(
            "XGB", processed_training_data, target, processed_testing_data
        )

        # keep the ML modeling results, to be used on different markets:
        # ML_modeling_results[category] = {'model_XGB': model_XGB, 'model_linear': model_linear, 'target_bias': target_bias, 'chosen_indices_XGB': chosen_indices_XGB, 'chosen_indices_linear': chosen_indices_linear, 'target_std': target_std, 'numeric_indices': numeric_indices, 'all_titles_of_features': all_titles_of_features}

        Chosen_features_XGBoost = all_titles_of_features[chosen_indices_XGB]
        Chosen_features_linear = all_titles_of_features[chosen_indices_linear]

        # Check overfitting of models that were created on reduced data
        result_XGB, result_linear = economic_model_ML_funcs.check_model_performance(
            selected_train_data_XGB,
            selected_train_data_linear,
            scores_reduced_XGB,
            scores_reduced_linear,
            model_XGB,
            model_linear,
            target,
        )
        print(result_XGB)
        print(result_linear)

        economic_model_ML_funcs.report_R_squared(
            target, scores_reduced_linear, scores_reduced_XGB
        )

        prediction_linear = economic_model_ML_funcs.make_predictions(
            "linear",
            target_bias,
            target_std,
            selected_test_data_linear,
            selected_test_data_XGB,
            model_XGB,
            model_linear,
        )
        prediction_XGB = economic_model_ML_funcs.make_predictions(
            "XGB",
            target_bias,
            target_std,
            selected_test_data_linear,
            selected_test_data_XGB,
            model_XGB,
            model_linear,
        )

        # add the predicted values to the dataframe dfm_2
        dfm_2.loc[
            (dfm_2.category == category)
            & (dfm_2.energy_source_code == pair[0])
            & (dfm_2.prime_mover_code == pair[1]),
            "fitting_param_intercept",
        ] = prediction_XGB

        # print the importance of each feature in the XGB model
        print(
            pd.Series(
                model_XGB.feature_importances_, index=Chosen_features_XGBoost
            ).sort_values(ascending=False)
        )

        # # print the importance of each feature in the linear model
        # print(pd.Series(model_linear.coef_, index = Chosen_features_linear).sort_values(ascending=False))

        # dfm_2.loc[dfm_2.category == category, 'fitting_param_intercept'] = prediction_XGB


dfm_2 = dfm_2_copy

####################################################### INTERCEPT (end)
####################################################### HEAT RATE SLOPE  (start)
# Calculation of HEAT RATE


# make a copy of dfm_2 that we will use to create the model
dfm_2_copy = dfm_2.copy()

# add a column to dfm_2 with the category, based on the results:
# dfm_2['category'] = dfm_2.apply(lambda row: results[(row.energy_source_code, row.prime_mover_code)], axis=1)
dfm_2["category"] = dfm_2.apply(
    lambda row: results.get((row.energy_source_code, row.prime_mover_code), "other"),
    axis=1,
)

# convert the value in the column 'operating_year' to 2024 - the value in the column 'operating_year'
dfm_2["operating_year"] = 2024 - dfm_2.operating_year

# read the file that contains the ML_modeling_results that was created using ERCOT data
with open("ML_modeling_results.pkl", "rb") as fdesc:
    ML_modeling_results = pickle.load(fdesc)

# eliminate the column "operating month", "latitude" and "longitude", eia_utility_id and eia_plant_code from the dataframe dfm_2
dfm_2 = dfm_2.drop(
    columns=[
        "operating_month",
        "latitude",
        "longitude",
        "eia_utility_id",
        "eia_plant_code",
        "plant_state",
        "plant_name",
    ]
)


# use the functions in economic_model_ML_funcs to predict pmin_median. This should be done for each category that appears more than 20 times in the dataframe
for category in ML_modeling_results.keys():
    raw_testing_data_ALL = dfm_2[dfm_2.category == category].drop(
        columns=["fitting_param_x"]
    )  # fitting_param_x
    raw_training_data_with_target_ALL = pd.read_csv(
        f"raw_training_data_with_target_{category}.csv"
    )

    # for each unique pair of 'energy_source_code' and 'prime_mover_code', find the number of rows in raw_testing_data_ALL that have that pair
    counts = raw_training_data_with_target_ALL.groupby(
        ["energy_source_code", "prime_mover_code"]
    ).size()

    # for each unique pair of 'energy_source_code' and 'prime_mover_code', if the number of rows in raw_testing_data_ALL if less than 20 then continue
    for pair in counts.index:
        print("number of points to work on: ", counts[pair])
        if counts[pair] < 20:
            continue

        # we want to work with each pair separately. Keep only the rows in raw_testing_data_ALL that have the pair
        raw_testing_data = raw_testing_data_ALL[
            (raw_testing_data_ALL.energy_source_code == pair[0])
            & (raw_testing_data_ALL.prime_mover_code == pair[1])
        ]
        raw_training_data_with_target = raw_training_data_with_target_ALL[
            (raw_training_data_with_target_ALL.energy_source_code == pair[0])
            & (raw_training_data_with_target_ALL.prime_mover_code == pair[1])
        ]

        # if the number of rows in raw_testing_data is 0, then continue
        if raw_testing_data.shape[0] == 0:
            continue

        # find columns in raw_testing_data that are not in raw_training_data_with_target
        # missing_columns = list(set(raw_testing_data.columns) - set(raw_training_data_with_target.columns))

        # Preprocess raw data, to impute NaN, scale numerical features and remove target bias
        (
            processed_training_data,
            processed_testing_data,
            target,
            target_bias,
            target_std,
            numeric_indices,
            all_titles_of_features,
        ) = economic_model_ML_funcs.preprocess(
            None,
            None,
            "fitting_param_x",
            raw_training_data_with_target,
            raw_testing_data,
            False,
        )
        if processed_training_data is None:
            continue

        # create models on reduced train datasets
        (
            selected_train_data_linear,
            model_linear,
            scores_reduced_linear,
            selected_test_data_linear,
            chosen_indices_linear,
        ) = economic_model_ML_funcs.build_model_on_reduced_data(
            "linear", processed_training_data, target, processed_testing_data
        )
        (
            selected_train_data_XGB,
            model_XGB,
            scores_reduced_XGB,
            selected_test_data_XGB,
            chosen_indices_XGB,
        ) = economic_model_ML_funcs.build_model_on_reduced_data(
            "XGB", processed_training_data, target, processed_testing_data
        )

        # keep the ML modeling results, to be used on different markets:
        # ML_modeling_results[category] = {'model_XGB': model_XGB, 'model_linear': model_linear, 'target_bias': target_bias, 'chosen_indices_XGB': chosen_indices_XGB, 'chosen_indices_linear': chosen_indices_linear, 'target_std': target_std, 'numeric_indices': numeric_indices, 'all_titles_of_features': all_titles_of_features}

        Chosen_features_XGBoost = all_titles_of_features[chosen_indices_XGB]
        Chosen_features_linear = all_titles_of_features[chosen_indices_linear]

        # Check overfitting of models that were created on reduced data
        result_XGB, result_linear = economic_model_ML_funcs.check_model_performance(
            selected_train_data_XGB,
            selected_train_data_linear,
            scores_reduced_XGB,
            scores_reduced_linear,
            model_XGB,
            model_linear,
            target,
        )
        print(result_XGB)
        print(result_linear)

        economic_model_ML_funcs.report_R_squared(
            target, scores_reduced_linear, scores_reduced_XGB
        )

        prediction_linear = economic_model_ML_funcs.make_predictions(
            "linear",
            target_bias,
            target_std,
            selected_test_data_linear,
            selected_test_data_XGB,
            model_XGB,
            model_linear,
        )
        prediction_XGB = economic_model_ML_funcs.make_predictions(
            "XGB",
            target_bias,
            target_std,
            selected_test_data_linear,
            selected_test_data_XGB,
            model_XGB,
            model_linear,
        )

        # add the predicted values to the dataframe dfm_2
        dfm_2.loc[
            (dfm_2.category == category)
            & (dfm_2.energy_source_code == pair[0])
            & (dfm_2.prime_mover_code == pair[1]),
            "fitting_param_x",
        ] = prediction_XGB

        # print the importance of each feature in the XGB model
        print(
            pd.Series(
                model_XGB.feature_importances_, index=Chosen_features_XGBoost
            ).sort_values(ascending=False)
        )

        # # print the importance of each feature in the linear model
        # print(pd.Series(model_linear.coef_, index = Chosen_features_linear).sort_values(ascending=False))

        # dfm_2.loc[dfm_2.category == category, 'fitting_param_x'] = prediction_XGB


dfm_2 = dfm_2_copy

####################################################### HEAT RATE SLOPE (end)

# dfm_3 = dfm_2.reset_index().rename(columns={'index': 'name'})
# dfm_3_offset = dfm_3[dfm_3['uid'].str.contains('OFFSET')]

# convert the value in the column 'operating_year' to 2024 - the value in the column 'operating_year'
# dfm_2['operating_year'] = 2024 - dfm_2.operating_year

# Michael ends here

# NOTE: in the table df_rdb, the column 'resource_type' is the same as 'energy_source_code' in the table dfm_2.
cnttt = 0

# list of problematic generators
# HOOKER_E MKTUNIT    this is the only one of the family of hookers that is online
# HOOKER_E OFFSET
# EXXMOB   MKTUNIT   this is the only one of the family of EXXON that is online
# EXXMOB   OFFSET
# COW      MKTUNIT
# COW      OFFSET
# CARVIL   GBC_MKT
# CARVIL   GA_MKT
# CARVIL   OFFSET_A
# CARVIL   OFFSET_BC
# EVRGRN   MKT_UNIT
# EVRGRN   OFF_SET
# STHSID   MKT_UN
# STHSID   OFFSET
# FORMOS   MKT_UN
# FORMOS   OFFSET_UN
# REP      MKT_UN
# REP      OFFSET_UN
# ENCO     MKTUNIT
# ENCO     OFFSET
# WODSTK   MKT_UN
# WODSTK   OFFSET_UN
# ESSO     MKTUNIT
# ESSO     OFFSET
# VFWPRK   OFFSET_UN
# VFWPRK   MKT_UN
# EXXON    OFFSET
# EXXON    MKTUNIT
# MCV      OFFSET
# PPG      GEN_OFFSET
# FRONTR   UN2_OFFSET
# FRONTR   UN1_OFFSET
# FRONTR   UN3_OFFSET
# DOWMTR   OFFSET


# UNCARB   OFFSET               I remove it, but there is no MKT or equivalent
# MONTPS   ST_MKT1              I keep it
# MONTPS   ST_MKT2              I keep it
# STCHAR   ST_MKT1              I keep it
# STCHAR   ST_MKT2              I keep it
# PPG                           I remove it. There is offset but instead of MKT there is Gen_axial!
# DOWMTR                        I remove the OFFSET, but there is no equivalent of MKT

# For each row in 'dfm_2', if the pair ('fitting_param_x', 'fitting_param_intercept') is too far from the group,
# replace the values of 'fitting_param_x' and 'fitting_param_intercept' with the average values of the group
# Otherwise, keep the original values of 'fitting_param_x' and 'fitting_param_intercept'
if False:
    counts = dfm_2.groupby(["prime_mover_code", "energy_source_code"]).size()

    avg_heat_rate = dfm_2.groupby(["prime_mover_code"]).agg(
        {
            "fitting_param_x": "median",
            "fitting_param_intercept": "median",
            "pmax_max": "max",
        }
    )

    dfm_2.loc[:, ["fitting_param_x", "fitting_param_intercept"]] = dfm_2.apply(
        lambda r: (
            avg_heat_rate.loc[(r.prime_mover_code)]
            if pd.isnull(r.fitting_param_x)
            else r[["fitting_param_x", "fitting_param_intercept"]]
        ),
        axis=1,
    )


counts = dfm_2.groupby(["prime_mover_code", "energy_source_code"]).size()
counts = dfm_2.groupby(["energy_source_code", "prime_mover_code"]).size()
# find the mean of fitting_param_x for each energy_source_code
dfm_2.groupby("energy_source_code")["fitting_param_x"].mean()

# dfm_2.groupby('energy_source_code')['fitting_param_x', 'fitting_param_intercept'].agg(['mean', 'count']).sort_values(('fitting_param_x', 'count'), ascending=False)
# dfm_2.groupby('energy_source_code', 'prime_mover_code')['fitting_param_x', 'fitting_param_intercept'].agg(['mean', 'count']).sort_values(('fitting_param_x', 'count'), ascending=False)
# dfm_2.groupby(['energy_source_code', 'prime_mover_code'])['fitting_param_x', 'fitting_param_intercept'].agg(['mean', 'count']).sort_values( 'energy_source_code', ascending=True)
dfm_2.groupby(["energy_source_code", "prime_mover_code"])[
    ["fitting_param_x", "fitting_param_intercept"]
].agg(["mean", "count"]).sort_values("energy_source_code", ascending=True)
dfm_2.groupby("energy_source_code")["fitting_param_x"].mean()


##################################################
# From Scott Bruns
df = pd.read_parquet("./output.parquet")
dfm = (
    df.groupby(["plant_id", "plant_name", "reported_fuel_type_code"])[
        ["total_fuel_consumption_mmbtu", "net_generation_mwh"]
    ]
    .sum()
    .reset_index()
)

# remove outliers
dfm = dfm[dfm["net_generation_mwh"] > 10]

# rough calculation of mean heat_rate
dfm["heat_rate"] = dfm.total_fuel_consumption_mmbtu / dfm.net_generation_mwh

dfm = dfm.dropna(subset=["heat_rate"])

# remove outliers:

# convert the values in dfm_2['eia_plant_code'] to integers
dfm_2["eia_plant_code"] = dfm_2["eia_plant_code"].astype(int)

# add to dfm a column, 'prime_mover_code', that is taken from dfm_2.
# It will be found by matching the columns dfm['reported_fuel_type_code', 'plant_name'] with the columns dfm_2['energy_source_code' , 'plant_name'] and the column dfm_2['prime_mover_code']
dfm = dfm.merge(
    dfm_2[["energy_source_code", "plant_name", "prime_mover_code", "eia_plant_code"]],
    left_on=["reported_fuel_type_code", "plant_name", "plant_id"],
    right_on=["energy_source_code", "plant_name", "eia_plant_code"],
    how="left",
).drop(columns=["energy_source_code", "eia_plant_code"])

# remove dupliate rows in dfm
dfm = dfm.drop_duplicates()

dfm["prime_mover_code"] = dfm["prime_mover_code"].fillna("not_known")


# Calculate the mean and standard deviation of the group
group_stats = dfm.groupby(["reported_fuel_type_code", "prime_mover_code"])[
    ["heat_rate"]
].agg(["median", "std"])
# rows where group_stats['std'] is NaN should be replaced with 0
group_stats["heat_rate", "std"] = group_stats["heat_rate", "std"].fillna(0)


# Define a function to replace outliers
def replace_outliers(row):
    median_x = group_stats.loc[
        (row["reported_fuel_type_code"], row["prime_mover_code"]),
        ("heat_rate", "median"),
    ]
    std_x = group_stats.loc[
        (row["reported_fuel_type_code"], row["prime_mover_code"]), ("heat_rate", "std")
    ]

    if abs(row["heat_rate"] - median_x) > 2 * std_x:
        row["heat_rate"] = median_x

    return row


# Apply the function to dfm_2

dfm_orig = dfm.copy()
dfm = dfm.apply(replace_outliers, axis=1)

# find rows in dfm that are different than their couterpart rows in dfm_orig and print those rows
dfm[dfm["heat_rate"] != dfm_orig["heat_rate"]]


# dfm's index should be dfm['plant_name']
dfm = dfm.set_index("plant_name")


################### Gas Hub - start ###################

generators_with_close_hubs = pd.read_csv("./generators_with_close_hubs.csv")


################### Gas Hub - end ###################


##################################################
test_CA = []
test_CT = []
test_GT = []
test_ST = []
cntr_of_generators_not_in_EIA = 0
for row in dfm_2.reset_index().itertuples():

    # if 'Brien'.lower() in row.uid.lower():
    #     print(18)
    # find if the row is in dfm based on the uid and energy_source_code. Take into account that the name in row and the name in dfm may be a bit different so use heuristics to match them
    if row.plant_name in dfm.index:
        # Ensure dfm.loc[row.plant_name, :] always returns a DataFrame
        lines = dfm.loc[[row.plant_name], :]
        for line in lines.itertuples():
            if line.reported_fuel_type_code.lower() == row.energy_source_code.lower():
                dfm_2.loc[row.uid, "fitting_param_x"] = line.heat_rate
                dfm_2.loc[row.uid, "fitting_param_intercept"] = 0.0
                break
        else:
            cntr_of_generators_not_in_EIA += 1
            if category_mapping.get(row.energy_source_code, "other") in {"gas", "coal"}:
                key = (row.energy_source_code, row.prime_mover_code)
                dfm_2.loc[row.uid, "fitting_param_x"] = _DEFAULT_PRICE.get(key, 50.0)
                dfm_2.loc[row.uid, "fitting_param_intercept"] = 0.0

        # This is for testing purpuses
        # key =  (row.energy_source_code, row.prime_mover_code)
        # if key[0] == 'NG' and key[1] == 'CA':
        #     test_CA.append(dfm_2.loc[row.uid, 'fitting_param_x'])
        # elif key[0] == 'NG' and key[1] == 'CT':
        #     test_CT.append(dfm_2.loc[row.uid, 'fitting_param_x'])
        # elif key[0] == 'NG' and key[1] == 'GT':
        #     test_GT.append(dfm_2.loc[row.uid, 'fitting_param_x'])
        # elif key[0] == 'NG' and key[1] == 'ST':
        #     test_ST.append(dfm_2.loc[row.uid, 'fitting_param_x'])

    # FRONTR UN1   OFF_SET
    if "FRONTR".lower() in row.uid.lower() and "un1".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    # FRONTR UN2   OFF_SET
    if "FRONTR".lower() in row.uid.lower() and "un2".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    # FRONTR UN3   OFF_SET
    if "FRONTR".lower() in row.uid.lower() and "un3".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    # GEN_AXIAL   OFF_SET
    if "GEN_AXIAL".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    # MCV_MCV   OFF_SET
    if "MCV_MCV".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    if "MCV".lower() in row.uid.lower() and "offset".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    # EVRGRN   OFF_SET
    if "OFF_SET".lower() in row.uid.lower():
        print(row.uid)
        cnttt += 1
        continue

    if (
        "OFFSET".lower() in row.uid.lower()
        and not "STCHAR".lower() in row.uid.lower()
        and not "MONTPS".lower() in row.uid.lower()
    ):
        print(row.uid)
        cnttt += 1
        continue

    if (
        "MKT".lower() in row.uid.lower()
        and not "STCHAR".lower() in row.uid.lower()
        and not "MONTPS".lower() in row.uid.lower()
    ):
        print(row.uid)
        cnttt += 1
        continue

    SET_MUST_RUN_TRUE = False
    if "COAL_CR".lower() in row.uid.lower():
        SET_MUST_RUN_TRUE = True

    if "MCV".lower() in row.uid.lower():
        SET_MUST_RUN_TRUE = True

    key = (row.energy_source_code, row.prime_mover_code)
    min_up_time, min_down_time, ramp_rate = _DEFAULT_PHYSICAL_CHARACTERISTICS.get(
        key, (10, 10, 9999)
    )

    startup_cost = _DEFAULT_START_UP_COST.get(key, 1000)

    pmax = row.pmax_99_perc + 0.01
    pmin = row.pmin_median
    # if row.energy_source_code in {"SUN", "WND", "MWH"}:
    if category_mapping.get(row.energy_source_code, "other") in {
        "wind",
        "solar",
        "hydro",
    }:
        pmin = 0.0

    assert pmax > pmin, row

    # replace two lines
    # Note: we should handle 'planned retirement' and 'planned operation year'
    # start_date = date(int(row.operating_year), min(int(row.operating_month), 12), 1)

    # Note: we should handle 'planned retirement' and 'planned operation year'
    assert not pd.isnull(row.operating_month) or not pd.isnull(
        row.planned_operation_month
    )

    if not pd.isnull(row.operating_month):
        start_date = date(
            int(row.operating_year), min(int(row.operating_month), 12), 1
        ) - timedelta(days=365)
    else:
        start_date = date(
            int(row.planned_operation_year), int(row.planned_operation_month), 1
        ) - timedelta(days=365)
    ### End replace two lines

    end_date = None
    if not pd.isnull(row.retirement_month):
        assert not pd.isnull(row.retirement_year)
        end_date = date(int(row.retirement_year), int(row.retirement_month), 1)

    storage_capacity = None
    # if row.energy_source_code == "MWH":
    #     storage_capacity = storage_capacity_by_uid.get(row.uid, pmax)

    # If we are dealing with a nat gas generator: provide information about the closest gas hub
    current_supplier_hub_data = None
    if row.eia_plant_code in generators_with_close_hubs.plant_id.values:
        #     print(generators_with_close_hubs[['plant_name','closest_gas_hub_name','closest_gas_hub_lat', 'closest_gas_hub_lon','current_supplier__hub_name', 'current_supplier__hub_lat',
        #    'current_supplier__hub_lon']])
        this_generator_data = generators_with_close_hubs[
            generators_with_close_hubs["plant_id"] == row.eia_plant_code
        ]
        # current_supplier_hub_name = this_generator_data['current_supplier__hub_name'].values[0]

        gas_hubs_and_coefs_for_this_generator = coefficients[row.plant_name]
        current_supplier_hub_data = {"coefficients": {}, "bias": None}
        for hub in gas_hubs_and_coefs_for_this_generator.iteritems():
            coef = hub[1]
            hub_name = hub[0]
            if coef > 0.001:
                # current_supplier_hub_data = {'coefficients': {'hub1':0.7, 'hub2':0.2, 'hub3':0.1}, 'bias': 3.0}   #example
                current_supplier_hub_data["coefficients"][hub_name] = coef
        current_supplier_hub_data["bias"] = intercepts.loc[row.plant_name].Value

    generators = [
        {
            "uid": row.uid,
            "state": row.plant_state,
            "station": "DUMMY",
            "kv": 1.0,
            "coordinates": {"latitude": row.latitude, "longitude": row.longitude},
            "eia_uid": {"eia_id": row.eia_plant_code, "unit_id": row.eia_gen_id},
            "psse_generator_uid": row.uid,
        }
    ]

    # Let's not think of CCs for now
    resource = {
        "uid": row.uid,
        "generators": generators,
        "energy_source_code": row.energy_source_code,
        "prime_mover_code": row.prime_mover_code,
        "physical_properties": {
            "pmin": pmin,
            "pmax": pmax,
            "min_up_time": min_up_time,
            "min_down_time": min_down_time,
            "ramp_rate": ramp_rate,
            "storage_capacity": storage_capacity,
        },
        "gas_hub": {
            # current_supplier_hub_data = {'coefficients': {'hub1':0.7, 'hub2':0.2, 'hub3':0.1}, 'bias': 3.0}
            "current_supplier_hub_data": current_supplier_hub_data  # zzz
        },
        "start_date": start_date,
        "end_date": end_date,
        # TO BE HANDLED
        "is_offns": True,
        "is_offqs": True,
        "must_run": SET_MUST_RUN_TRUE,
    }

    # Let's generate the propre curve with federal credit etc...
    # if row.energy_source_code in {"WND", "SUN"}:
    if category_mapping.get(row.energy_source_code, "other") in {
        "wind",
        "solar",
        "hydro",
        "nuclear",
    }:
        offer_curve = {
            "type": "DIRECT",
            "min_gen_cost": _DEFAULT_PRICE.get(key, 0.0),
            "blocks": [{"quantity": pmax, "price": _DEFAULT_PRICE.get(key, 0.0)}],
        }
    # elif row.energy_source_code in {"NG", "SUB", "LIG"}:
    elif category_mapping.get(row.energy_source_code, "other") in {"gas", "coal"}:
        assert not pd.isnull(row.fitting_param_x)
        assert not pd.isnull(row.fitting_param_intercept)
        offer_curve = {
            "type": "HEAT_RATE_BASED",
            "coef": dfm_2.loc[row.uid, "fitting_param_x"],
            "intercept": dfm_2.loc[row.uid, "fitting_param_intercept"],
        }
    else:
        # if not key in [('RFO','ST'), ('DFO','GT'), ('DFO','ST'), ('PUR', "ST"), ('BFG', 'ST'), ('DFO','CT'), ('KER', 'GT'), ('OG', 'OT'), ('DFO', 'IC'), ("MSW", "ST"), ('BLQ', 'ST'), ('OG', 'ST'), ('WDS', 'ST'), ('WH', 'ST'), ('LFG', 'ST'), ('OG', 'CA'), ('LFG', 'IC'), ('LFG', 'CT'), ('DFO', 'CA'), ('LFG', 'GT'), ('OBG', 'IC'), ('JF', 'GT'), ('WH', 'OT') ]:
        #     print(18)
        offer_curve = {
            "type": "DIRECT",
            "min_gen_cost": _DEFAULT_PRICE.get(key, 50.0),
            "blocks": [{"quantity": pmax, "price": _DEFAULT_PRICE.get(key, 50.0)}],
        }

    supply_curves[row.uid] = {
        "start_up_cost": {"type": "DIRECT", "value": startup_cost},
        "offer_curve": offer_curve,
    }

    resources.append(resource)

# here we run on plants that were not in Scott Bruns' list:
if True:
    # Calculate the mean and standard deviation of the group
    # group_stats = dfm_2.groupby(['energy_source_code', 'prime_mover_code'])[['fitting_param_x', 'fitting_param_intercept']].agg(['median', 'std'])
    group_stats = (
        dfm_2[dfm_2.fitting_param_x > 0.0]
        .groupby(["energy_source_code", "prime_mover_code"])[
            ["fitting_param_x", "fitting_param_intercept"]
        ]
        .agg(["median", "std"])
    )

    # Define a function to replace outliers
    def replace_outliers(row):
        # check if (row['energy_source_code'], row['prime_mover_code']) is in group_stats
        if (
            row["energy_source_code"],
            row["prime_mover_code"],
        ) not in group_stats.index:
            return row

        median_x = group_stats.loc[
            (row["energy_source_code"], row["prime_mover_code"]),
            ("fitting_param_x", "median"),
        ]
        std_x = group_stats.loc[
            (row["energy_source_code"], row["prime_mover_code"]),
            ("fitting_param_x", "std"),
        ]
        median_intercept = group_stats.loc[
            (row["energy_source_code"], row["prime_mover_code"]),
            ("fitting_param_intercept", "median"),
        ]
        std_intercept = group_stats.loc[
            (row["energy_source_code"], row["prime_mover_code"]),
            ("fitting_param_intercept", "std"),
        ]

        if (
            abs(row["fitting_param_x"] - median_x) > 2 * std_x
            or abs(row["fitting_param_intercept"] - median_intercept)
            > 2 * std_intercept
        ):
            row["fitting_param_x"] = median_x
            row["fitting_param_intercept"] = median_intercept

        if median_x != 0.0 and row["fitting_param_x"] == 0.0:
            row["fitting_param_x"] = median_x
            row["fitting_param_intercept"] = median_intercept

        return row

    # Apply the function to dfm_2
    dfm_2 = dfm_2.apply(replace_outliers, axis=1)

if False:
    # plot a histogram of the heat rate of all the dfm_2 rows where the energy_source_code is 'NG' and the prime_mover_code is 'GT'
    dfm_2[
        (dfm_2.energy_source_code == "NG") & (dfm_2.prime_mover_code == "GT")
    ].fitting_param_x.hist(bins=50)
    plt.title("Heat Rate of NG GT")
    plt.figure()

    # plot a histogram of the heat rate of all the dfm_2 rows where the energy_source_code is 'NG' and the prime_mover_code is 'GT'
    dfm_2[
        (dfm_2.energy_source_code == "NG") & (dfm_2.prime_mover_code == "CA")
    ].fitting_param_x.hist(bins=50)
    plt.title("Heat Rate of NG CA")
    plt.figure()

    # plot a histogram of the heat rate of all the dfm_2 rows where the energy_source_code is 'NG' and the prime_mover_code is 'GT'
    dfm_2[
        (dfm_2.energy_source_code == "NG") & (dfm_2.prime_mover_code == "CT")
    ].fitting_param_x.hist(bins=50)
    plt.title("Heat Rate of NG CT")
    plt.figure()

    # plot a histogram of the heat rate of all the dfm_2 rows where the energy_source_code is 'NG' and the prime_mover_code is 'GT'
    dfm_2[
        (dfm_2.energy_source_code == "NG") & (dfm_2.prime_mover_code == "ST")
    ].fitting_param_x.hist(bins=50)
    plt.title("Heat Rate of ST ")

dfm_2.groupby(["energy_source_code", "prime_mover_code"])[
    ["pmin_median", "pmax_99_perc"]
].agg(["median", "max", "min", "count"])

plt.show()

dfsc = pd.Series(supply_curves)
dfsc[dfsc.index.str.contains("VOGTLE1")].iloc[0]


# In[ ]:

print(f"Number of OFFSET and MKT removed: {cnttt}")

version = "resourcedb-v4/spp/20240423t1825"

# jupyter: run the following.
# get_ipython().system('mkdir -p $version')
# VS-Code: run the following.
os.makedirs(version, exist_ok=True)

with open(f"{version}/resources.json", "w") as fdesc:
    fdesc.write(jsonplus.dumps(resources))

# create dataframe from resources
df_resources_test = pd.DataFrame(resources)


# in resources we want to find the one where uid is 'COAL_CR':
# for r in resources:
#     if 'MCV' in r["uid"]:
#         print(r)


#######
# Check if the supply_curves dictionary is serializable
# def find_unserializable_item(supply_curves):
#     for key, value in supply_curves.items():
#         try:
#             # Attempt to serialize the item
#             json.dumps(value)
#         except TypeError as e:
#             # If serialization fails, print the key and the error
#             print(f"Serialization failed for key: {key}, Error: {e}")
#             # Optionally, return or break if you want to stop at the first failure
#             # return key, value
#     print("All items are serializable")

# find_unserializable_item(supply_curves)


with open(f"{version}/supply_curves.json", "w") as fdesc:
    fdesc.write(jsonplus.dumps(supply_curves))


######################
# Find missing generators

if False:
    # the two sources for the parquet files:
    # "s3://enverus-pr-ue1-cdr-unrestricted-mu-ingested-prod/crcl/hrrr-solar-forecasts/20240129T2045Z/2024-05-06T06:00:00+00:00.parquet"
    # "s3://enverus-pr-ue1-cdr-unrestricted-mu-ingested-prod/crcl/hrrr-wind-forecasts/20240129T2045Z/2024-05-06T06:00:00+00:00.parquet"
    df_crcl_wind = pd.read_parquet("2024-05-06T06:00:00+00:00_wind.parquet")
    df_crcl_solar = pd.read_parquet("2024-05-06T06:00:00+00:00_solar.parquet")
    df_crcl_wind = df_crcl_wind[df_crcl_wind["iso"] == "SPP"]
    df_crcl_solar = df_crcl_solar[df_crcl_solar["iso"] == "SPP"]

    # put both dataframes on under the other in a new dataframe
    df_crcl = pd.concat([df_crcl_wind, df_crcl_solar], ignore_index=True)

    # remove from df_crcl rows where 'status' is in ['retired', 'planned', 'built_not_in_operation', 'under_construction', 'cancelled', 'proposed'].
    # Take care of a situation that 'status' may be a string in small or capital letter
    string_to_use_lower = "retired|planned|built_not_in_operation|under_construction|cancelled|proposed".lower()
    string_to_use_upper = "retired|planned|built_not_in_operation|under_construction|cancelled|proposed".upper()
    df_crcl = df_crcl[~df_crcl["status"].str.contains(string_to_use_lower)]
    df_crcl = df_crcl[~df_crcl["status"].str.contains(string_to_use_upper)]

    # df_crcl = df_crcl[~df_crcl['status'].str in ['retired', 'planned', 'built_not_in_operation', 'under_construction', 'cancelled', 'proposed']]
    df_crcl = df_crcl.groupby(["location", "env_id", "eia_id"])[
        ["eia_id", "location", "capacity_mw"]
    ].first()

    # dfm_2 = dfm_2.groupby(['eia_utility_id', 'eia_plant_code', 'eia_gen_id'])[['eia_utility_id']].first()
    dfm_2 = dfm_2.groupby(["eia_utility_id", "eia_plant_code", "eia_gen_id"])[
        ["eia_plant_code", "plant_name"]
    ].first()

    # join df_crcl with dfm_2 to find the generators that are in df_crcl but not in dfm_2
    dfm_2["uid"] = dfm_2.eia_plant_code
    dfm_2 = dfm_2.reset_index(drop=True)
    df_crcl["uid"] = df_crcl["eia_id"]
    df_crcl["uid"] = pd.to_numeric(df_crcl["uid"], errors="coerce")
    df_crcl = df_crcl.reset_index(drop=True)
    df_missing = pd.merge(df_crcl, dfm_2, on="uid", how="outer", indicator=True)

    df_exists_in_both = df_missing[df_missing["_merge"] == "both"]
    df_exists_only_in_dfm_2 = df_missing[df_missing["_merge"] == "right_only"]
    df_exists_only_in_df_crcl = df_missing[df_missing["_merge"] == "left_only"]

    # find rows from df_exists_only_in_df_crcl that are not in dfm_original__, based on the uid
    df_exists_only_in_df_crcl = df_exists_only_in_df_crcl.set_index("uid")
    df_exists_only_in_df_crcl = df_exists_only_in_df_crcl[
        ~df_exists_only_in_df_crcl.index.isin(dfm_original__.index)
    ]

    df_exists_only_in_df_crcl.to_csv("SPP2_df_exists_only_in_df_crcl.csv", index=False)

    dfm_original__["uid"] = dfm_original__.eia_plant_code
    df_exists_in_dfm_original__but_not_dfm_2 = dfm_original__[
        ~dfm_original__.uid.isin(dfm_2.uid)
    ].plant_name.unique()

    # print the generators that are in df_crcl but not in dfm_2 (i.e., in CRCL but not in our system)
    print(
        df_exists_only_in_df_crcl[["capacity_mw", "location"]]
        .query("capacity_mw > 50")
        .sort_values("capacity_mw", ascending=False)
        .sort_values("capacity_mw", ascending=False)
    )
    print(
        df_exists_in_both.shape,
        df_exists_only_in_dfm_2.shape,
        df_exists_only_in_df_crcl.shape,
    )

    print("------------------------------------------------")
    # #check mismatching names between EIA and our data:
    # for line in df_exists_in_both.itertuples():
    #     # plot the plant_name and the location of the generator and format the output to be more readable
    #     print(line.plant_name,'     |      ', line.location)

    # print(line.plant_name, line.location)

    print(18)
    if False:
        get_ipython().system("ls resourcedb-v4/miso/20240423")

        resources[5]

        get_ipython().system("cd resourcedb-v4/miso/20240423 && find $(pwd)")

        df = pd.read_csv("/Users/michael.simantov/Desktop/portfolio/miso_ctg_v2.csv")

        df = pd.DataFrame(resources)

        [
            r
            for r in resources
            if r["physical_properties"]["pmin"] <= r["physical_properties"]["pmax"]
        ]

        df[df.uid == "FERGUS2  STILLWATER_EC2"].physical_properties.iloc[0]

        df = pd.read_parquet(
            "/Users/michael.simantov/Documents/draft/ps_engine_results/miso/first-20240423/denormalized_hub/08204d01-85a9-45ec-8226-692b610ac457 -- 2024-04-06T04:00:00-05:00 -- 2024-04-06T02:00:00-05:00.parquet"
        )

        df.timestamp = df.timestamp.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

        df[df.name == "ILLINOIS.HUB"].set_index("timestamp").price.plot(grid=True)
        plt.ylim(0, None)

        set(df["Contingency Name"].unique())

    # %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from models.extract_csv import read_datafiles

data_dict = read_datafiles()

df_local = data_dict["local_movement_data"]
df_travel_matrices = data_dict["monthly_travel_matrices"]
df_pop = data_dict["population_data"]
df_epi = data_dict["covid_data"]

# todo: duplicated definition
origin_date = pd.to_datetime("2020-01-01")  # Day corresponding to t = 0

df_epi["day"] = (pd.to_datetime(df_epi["isoweek"]) - origin_date).dt.days

# Read NYC epi data (separate file)
df_epi_nyc = pd.read_csv("./data/toby.epi.nyc.csv")
df_epi_nyc["isoweek"] = pd.to_datetime(df_epi_nyc["isoweek"])
df_epi_nyc["day"] = (pd.to_datetime(df_epi_nyc["isoweek"])
                     - origin_date).dt.days
nyc_fips = [36005, 36047, 36061, 36081, 36085]
nyc_pop = df_pop[df_pop["fips"].isin(nyc_fips)]["pop"].sum()

# Counties to plot
counties = ["New York County, New York",
            "Kings County, New York",
            # "Los Angeles County, California",
            # "King County, Washington",
            # "DeKalb County, Georgia",
            # "Gallatin County, Montana",
            # "Clarke County, Georgia",
            # "Knox County, Ohio",
            # "Fairbanks North Star Borough, Alaska",
            # "Hardin County, Kentucky"
            ]


df_simulated = pd.read_csv("./output_data/simulated_new_cases.csv",
                           index_col=0)
df_simulated_full = df_simulated.copy()

counties = df_simulated.columns[:-1]

#%%
# Note: this is hacky and needs refactoring
df_quantiles = (df_simulated_full.groupby(level=0)
                .apply(lambda x: x.quantile(q=[0.025, 0.975]))).swaplevel()

plot_run = 0
df_simulated = df_simulated[df_simulated["run"] == 0]
fig, axes = plt.subplots(nrows=3, figsize=(6, 6))

for county in counties:

    ax = axes[0]
    county_pop = df_pop[df_pop["name"] == county]["pop"].iloc[0]
    # ax.plot(df_plot.index, 1e5 * df_plot[county] / county_pop,
    #         label=county.replace(" County", "").replace(" Borough", ""))
    ax.plot(df_simulated.index, 1e5 * df_simulated[county] / county_pop,
            label=county.replace(" County", "").replace(" Borough", ""))
    ax.fill_between(df_simulated.index,
                    1e5 * df_quantiles.loc[0.025][county] / county_pop,
                    1e5 * df_quantiles.loc[0.975][county] / county_pop,
                    alpha=0.5)
    ax = axes[1]
    fips = df_pop[df_pop["name"] == county]["fips"].iloc[0]
    time_series = df_epi[df_epi["fips"] == fips].copy()

    ax.plot(time_series["day"], 1e5 * time_series["deaths"] / county_pop,
            label=county.replace(" County", ""))

    ax = axes[2]

    fips = df_pop[df_pop["name"] == county]["fips"].iloc[0]
    time_series = df_epi[df_epi["fips"] == fips].copy()
    ax.plot(time_series["day"], 1e5 * time_series["cases"] / county_pop,
            label=county.replace(" County", ""))

# axes[1].plot(df_epi_nyc["day"], 1e5 * df_epi_nyc["deaths"] / nyc_pop,
#              color="black")
# axes[2].plot(df_epi_nyc["day"], 1e5 * df_epi_nyc["cases"] / nyc_pop,
#              color="black")


ax = axes[0]
ax.set_ylabel("Simulated deaths\nper $10^5$")
#ax.set_yscale("log")
ax.legend(loc=(0.2, 0.6), frameon=False, ncol=2, handlelength=1)

ax = axes[1]
ax.set_ylabel("Weekly deaths\nper $10^5$")
#ax.set_yscale("log")
ax.set_xlabel("Day")

ax = axes[2]
ax.set_ylabel("Weekly confirmed\ncases per $10^5$")
#ax.set_yscale("log")
ax.set_xlabel("Day")

abc_list = list("ABCDEFGHIJ")

for i, ax in enumerate(axes):
    ax.axvline((pd.to_datetime("2020-03-01") - origin_date).days,
               linestyle="--", color="grey")
    ax.set_ylim(0., 10 if i < 2 else 200)
    ax.set_title(abc_list[i], loc="left")
    ax.set_xlim(0, 365)
    sns.despine(ax=ax, trim=True)



fig.subplots_adjust(left=0.15, right=0.99, top=0.94,
                    bottom=0.1, hspace=0.5, wspace=0.25)
fig.savefig("./figures/fig_test_variable_matrix_4.pdf")
plt.show()


#%% Calculate first day threshold exceeded

def get_first_day(x, threshold=10):
    try:
        week_number = x[x > threshold].index[0]
    except IndexError:
        week_number = np.nan
    return week_number

threshold = 5

# Simulated first day
df_first_day = (df_simulated.apply(get_first_day, threshold=threshold, axis=0)
                .to_frame())
df_first_day.columns = ["Simulated first day"]

df_first_day[["fips", "state", "pop"]] = (df_pop.set_index("name")
                        .loc[df_first_day.index, ["fips", "state", "pop"]])


# Data first day
df_first_day = df_first_day.reset_index().set_index("fips")
epi_data_first_week = (df_epi[-df_epi["fips"].isna()]
                       .set_index(["fips", "day"])["cases"]
                       .unstack(level=0)
                       .fillna(0)
                       .apply(get_first_day, threshold=threshold, axis=0))

# Drop missing counties (includes New York city)
df_first_day = df_first_day[df_first_day.index.isin(epi_data_first_week.index)]
df_first_day["Data first day"] = epi_data_first_week.loc[df_first_day.index]

# Calculate difference between data and simulated
df_first_day["Difference"] = (df_first_day["Simulated first day"]
                              - df_first_day["Data first day"])

#%%
fig, axes = plt.subplots(ncols=3, figsize=(10, 4))

ax = axes[0]
ax.scatter(df_first_day["Data first day"],
           df_first_day["Simulated first day"])

test = df_first_day[df_first_day["state"] == 15]
ax.scatter(test["Data first day"],
           test["Simulated first day"])
ax.plot([25, 100], [0, 75], color="black")
ax.set_ylabel("First day simulated cases > " + str(threshold))

ax.set_xlabel("First day reported cases > " + str(threshold))
ax.set_xlim(0, 365)
ax.set_ylim(0, 200)

ax = axes[1]
ax.scatter(df_first_day["Simulated first day"],
           df_first_day["pop"])

ax.set_ylabel("Population size")

ax.set_xlabel("Simulated first day")
ax.set_yscale("log")

ax = axes[2]
ax.scatter(df_first_day["Difference"],
           df_first_day["pop"])

# test = df_first_day[df_first_day["state"] == 28]
# ax.scatter(test["Difference"],
#            test["pop"])

ax.set_ylabel("Population size")

ax.set_xlabel("Difference in first day")
ax.set_yscale("log")
for i, ax in enumerate(axes):
    sns.despine(ax=ax)
    ax.set_title(abc_list[i], loc="left", pad=-0)

fig.subplots_adjust(left=0.06, right=0.99, top=0.94,
                    bottom=0.12, hspace=0, wspace=0.25)

fig.savefig("./figures/fig_test_scatter_first_day.pdf")
plt.show()

#%% Plot maps for first day exceeding threshold
import geopandas as gpd

shape_path = "data/USA_Counties-shp/USA_Counties.shp"

df_shapes = gpd.read_file(shape_path)
df_shapes.index = df_shapes["FIPS"].astype(int)

map_plot_columns = ["Data first day", "Simulated first day", "Difference"]
for col in map_plot_columns:
    df_shapes[col] = df_first_day[col]

fig, axes = plt.subplots(ncols=3, figsize=(10, 3))
for i, ax in enumerate(axes):
    col = map_plot_columns[i]
    df_shapes.plot(ax=ax,
                   column=col,
                   #edgecolor="#FFFFFF",
                   linewidth=0.05,
                   categorical=False,
                   cmap="PuOr" if i == 2 else "viridis",
                   legend=True,
                   vmin=-200 if i == 2 else 0,
                   vmax=200 if i == 2 else 365,
                   missing_kwds=dict(color="lightgrey", ),
                   legend_kwds={'orientation': "horizontal",
                                "shrink": 0.5,
                                "label": col})
    # ax.set_xlim(-1.4e7, -0.74e7)
    # ax.set_ylim(0.28e7, 0.64e7)
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xlim(-126, -66.5)
    ax.set_ylim(24, 50)
    ax.set_title(abc_list[i], loc="left", pad=-0)
fig.subplots_adjust(left=0., right=0.99, top=0.94,
                    bottom=0.05, hspace=0, wspace=0.05)
fig.savefig("./figures/fig_first_day_maps_1.png", dpi=600)
plt.show()


#%% Plot local distancing data
from matplotlib import dates as mdates
fig, axes = plt.subplots()

ax = axes
for county in counties:
    fips = df_pop[df_pop["name"] == county]["fips"].iloc[0]
    df_plot = df_local[df_local["county"] == fips]
    ax.scatter(df_plot.index, df_plot["pr_local_scaled"])
    ax.plot(df_plot.index, df_plot["pr_local_scaled"], label=county)

# ax.scatter(np.arange(1, 13), local_tensor[:, seed_county])
# ax.plot(np.arange(1, 13), local_tensor[:, seed_county])
ax.legend(ncols=2, fontsize=8, frameon=False, handlelength=1)
ax.set_ylabel("Relative pr_local")
ax.set_xlabel("Month")
ax.set_xlim(pd.to_datetime(["2020", "2021"]))
locator = mdates.MonthLocator()
formatter = mdates.ConciseDateFormatter(locator, show_offset=False)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

#ax.set_xticks(np.arange(1, 13))
sns.despine(trim=True)
plt.show()
fig.savefig("./figures/fig_pr_local.pdf")


import warnings
import pandas as pd


def read_datafiles():
    covid_data_filename = "toby.covid.csv.gz"
    dist_data_filename = "toby.county.dist.csv.gz"
    dhc_data_filename = "toby.county.dhc.csv.gz"
    travel_data_filename = "toby.np-travel.csv.gz"
    local_movement_filename = "toby.np-co.csv.gz"
    distance_filename = "toby.county.dist.csv.gz"

    df_covid = pd.read_csv('data/' + covid_data_filename, compression="gzip")
    df_dist = pd.read_csv('data/' + dist_data_filename, compression="gzip")
    df_dhc = pd.read_csv('data/' + dhc_data_filename, compression="gzip")
    df_travel = pd.read_csv('data/' + travel_data_filename, compression="gzip")
    df_local = pd.read_csv('data/' + local_movement_filename,
                           compression="gzip")

    df_geo_dist = _read_distance_matrix(distance_filename)

    df_pop = df_dhc.set_index("name")

    df_local = _compute_local_movement_dataframe(df_local)

    df_pop = df_pop.sort_values("fips")

    # 2261 missing from pop data (Valdez–Cordova Census Area, Alaska)
    df_pop = df_pop[-df_pop["state"].isin([2, 15])]  # Drop Alaksa and Hawaii

    # Get set of FIPS codes present in both demographic and movement data
    fips_set = set(df_local["county"].unique()) & set(df_pop["fips"])

    df_pop = df_pop[df_pop["fips"].isin(fips_set)].reset_index()
    df_local = df_local[df_local["county"].isin(fips_set)]

    df_travel = df_travel.set_index(["from", "to"])
    df_travel["month_duration"] = (pd.to_datetime(df_travel["month"])
                                   .dt.daysinmonth)
    # Calculate movement rate (ratio of pr_orig_res to days in month)
    df_travel["movement_rate"] = (df_travel["pr_orig_res"] /
                                  df_travel["month_duration"])

    df_travel_matrices = (df_travel
                          .groupby("month")["movement_rate"]
                          .apply(_get_single_month_travel_matrix,
                                 fips_list=df_pop["fips"]))


    return {"population_data": df_pop,
            "covid_data": df_covid,
            "distance_data": df_dist,
            "travel_data": df_travel,
            "monthly_travel_matrices": df_travel_matrices,
            "local_movement_data": df_local,
            "geographical_distance": df_geo_dist}



def _get_single_month_travel_matrix(df, fips_list):

    df_single_month = df.unstack(level=1)
    # Fill missing rows and columns
    df_single_month = df_single_month.reindex(fips_list)
    df_single_month = df_single_month.reindex(fips_list, axis=1)
    df_single_month = df_single_month.fillna(0)
    return df_single_month


def _read_distance_matrix(dist_data_filename):

    # Read gzip archive
    df_dist = pd.read_csv('data/' + dist_data_filename, compression="gzip")

    # Data provided as long form upper triangle
    df_dist = df_dist.set_index(["fips1", "fips2"])["dist_km"].unstack(level=1)

    # Add first column and last row. Note this is not currently robust
    # to changes in the order of input data
    df_dist.loc[1001, 1001] = 0
    df_dist.loc[78030, 78030] = 0
    df_dist = df_dist.sort_index(axis=1).sort_index(axis=0)

    # Check it's a square matrix
    if not all(df_dist.index == df_dist.columns):
        warnings.warn("Matrix is not square")

    # Fill missing elements with 0
    df_dist = df_dist.fillna(0)

    # Fill lower triangle with transposed upper triangle
    df_dist = df_dist + df_dist.transpose()

    return df_dist


def _compute_local_movement_dataframe(df):
    df_local = df.copy()
    df_local["month"] = pd.to_datetime(df_local["month"])
    df_local = df_local.set_index(["county", "month"])
    df_local["pr_local_scaled"] = (df_local
                                   .groupby(level=0)["pr_local"]
                                   .apply(lambda x: x/x.iloc[0])
                                   .droplevel(0)
                                   .subtract(1)
                                   )
    # Only keep groups with full 12 months of data
    df_local = (df_local.groupby(level=0)
                .filter(lambda x: x["pr_local_scaled"].count() == 12))

    df_local = df_local.sort_index().reset_index(level=0)
    return df_local


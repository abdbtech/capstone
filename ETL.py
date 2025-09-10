# import libraries
# NOTE: global_vars should be edited to include local paths and credentials before use.
# If global_vars.py is created in the root dir remove the ignore/ prefix in the import statement below.
import ignore.global_vars as gv
import db_tools as dbt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import urllib.request
import re



# Load NOAA storm events and create database table

dbt.browse_ftp("ftp://ftp.ncei.noaa.gov", "/pub/data/swdi/stormevents/csvfiles/")

# Get file list where type is StormEvents_details and year is 1999-2025
files = dbt.get_ftp_filenames(
    "ftp://ftp.ncei.noaa.gov", "/pub/data/swdi/stormevents/csvfiles/"
)

# Filter for StormEvents_details files from 1999-2025
pattern = r"StormEvents_details-ftp_v1\.0_d(\d{4})_c.*\.csv\.gz"
selected_files = []

for file in files:
    match = re.match(pattern, file)
    if match:
        year = int(match.group(1))
        if 1999 <= year <= 2025:
            selected_files.append(file)

# get all files identified in 'filenames' and populate a df for cleaning

base_url = "ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
all_storm_data = []

for i, filename in enumerate(selected_files, 1):
    try:
        # Construct full URL
        full_url = base_url + filename

        # Stream file to DataFrame
        df = dbt.ftp_to_df(full_url, compression="gzip")

        if not df.empty:
            # Add year column for reference
            year = re.search(r"d(\d{4})", filename).group(1)
            df["FILE_YEAR"] = int(year)

            all_storm_data.append(df)
            print(f"{i:2d}/{len(selected_files)}: {year} - {len(df)} rows")
        else:
            print(f"{i:2d}/{len(selected_files)}: {filename} - No data")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Concatenate all DataFrames
if all_storm_data:
    df_all_storms = pd.concat(all_storm_data, ignore_index=True)
    print(
        f"\nCombined DataFrame: {len(df_all_storms)} total rows, {len(df_all_storms.columns)} columns"
    )
    print(
        f"Years covered: {df_all_storms['FILE_YEAR'].min()} - {df_all_storms['FILE_YEAR'].max()}"
    )

    # Show basic info
    df_all_storms.head()
else:
    print("No data was successfully loaded")

# Drop unneeded columns

df_all_storms_drop = df_all_storms[
    [
        "BEGIN_YEARMONTH",
        "BEGIN_DAY",
        "EPISODE_ID",
        "EVENT_ID",
        "EVENT_TYPE",
        "CZ_FIPS",
        "STATE_FIPS",
        "INJURIES_DIRECT",
        "INJURIES_INDIRECT",
        "DEATHS_DIRECT",
        "DEATHS_INDIRECT",
        "DAMAGE_PROPERTY",
    ]
]
df_all_storms_drop.head()

# combine BEGIN_YEARMONTH and BEGIN_DAY into a single DATE column and convert to datetime, drop original columns

df_all_storms_comb = df_all_storms_drop.copy()

df_all_storms_comb["BEGIN_YEARMONTH"] = df_all_storms_comb["BEGIN_YEARMONTH"].astype(
    str
)
df_all_storms_comb["BEGIN_DAY"] = (
    df_all_storms_comb["BEGIN_DAY"].astype(str).str.zfill(2)
)
df_all_storms_comb["DATE"] = (
    df_all_storms_comb["BEGIN_YEARMONTH"] + df_all_storms_comb["BEGIN_DAY"]
)
df_all_storms_comb["DATE"] = pd.to_datetime(df_all_storms_comb["DATE"], format="%Y%m%d")
df_all_storms_comb.drop(columns=["BEGIN_YEARMONTH", "BEGIN_DAY"], inplace=True)
df_all_storms_comb.head()

# combine state and county fips into a single high level FIPS. handle NA with convention of 99999 as unknown county
# keep original columns in case needed.
df_all_storms_comb["STATE_FIPS"] = (
    pd.to_numeric(df_all_storms_comb["STATE_FIPS"], errors="coerce")
    .fillna(99)
    .astype(int)
    .astype(str)
    .str.zfill(2)
)
df_all_storms_comb["CZ_FIPS"] = (
    pd.to_numeric(df_all_storms_comb["CZ_FIPS"], errors="coerce")
    .fillna(999)
    .astype(int)
    .astype(str)
    .str.zfill(3)
)
df_all_storms_comb["CO_FIPS"] = (
    df_all_storms_comb["STATE_FIPS"] + df_all_storms_comb["CZ_FIPS"]
)

# load dfs to db
dbt.load_data(df_all_storms_comb, "NOAA_STORM_EVENTS", if_exists="replace")


# Load 2023 census data and create database table

df = pd.read_csv(gv.DATA_PATHS["census_resilience"], encoding="latin-1")

# cast TRACT to string
df_tract = df.copy()
df_tract["TRACT"] = df_tract["TRACT"].astype(str)

# Drop unneeded columns
df_tract_only = df_tract_only.drop(
    columns=["STATE", "COUNTY", "NAME", "GEO_LEVEL", "TRACT", "WATER_TRACT"],
    errors="ignore",
)

# Create FIPS column from GEO_ID and drop original GEO_ID column
df_tract_only["FIPS"] = df_tract_only["GEO_ID"].str.split("US").str[1]
df_clean = df_tract_only.drop(columns=["GEO_ID"], errors="ignore")

# load to db when ready
dbt.load_data(df_clean, "census_resilience", if_exists="replace")
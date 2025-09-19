# import libraries
# NOTE: global_vars should be edited to include local paths and credentials before use.
# If global_vars.py is created in the root dir remove the ignore/ prefix in the import statement below.
import os
import re
import zipfile
import pandas as pd
import geopandas as gpd
import ignore.global_vars as gv
import db_tools as dbt


if __name__ == "__main__":

    '''
    NRI Shapefile ETL
    '''

    # Database connection setup
    dbt.connection(test=True)
    dbt.engine()

    zip_path = gv.DATA_PATHS["nri_shapefile"]
    extract_dir = gv.DATA_PATHS["extract_dir"]

    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Extract .shp file path
    shp_path = os.path.join(extract_dir, "NRI_Shapefile_CensusTracts.shp")

    # Load shapefile
    gdf = gpd.read_file(shp_path)

    gdf.to_postgis(name="nri_shape_census_tracts", con=dbt.engine(), if_exists="replace")

    '''
    Census Data ETL
    '''

    df = pd.read_csv(gv.DATA_PATHS["census_resilience"], encoding='latin-1')


    # Looking at the data there are missing leading 0 in the TRACT field, instead of concatonating
    # filter so only GEO_LEVEL == 'County' remain and strip everything except the last 5 digits from GEO_ID
    # into a new column County_fips
    df = df.copy()
    df = df[df['GEO_LEVEL'] == 'County']
    df = df.reset_index(drop=True)
    df['County_fips'] = df['GEO_ID'].str[-5:]

    # Select relevant columns
    df = df[
        [
            "County_fips",
            "GEO_ID",
            "GEO_LEVEL",
            "WATER_TRACT",
            "POPUNI",
            "PRED12_PE",
            "PRED3_PE",
        ]
    ]
    # Load into db
    dbt.load_data(df, "census_resilience", if_exists="replace")

    # Get file list where type is StormEvents_details and year is 1999-2024. 
    # 2025 excluded as it is incomplete as of 10SEP2025
    files = dbt.get_ftp_filenames("ftp://ftp.ncei.noaa.gov", "/pub/data/swdi/stormevents/csvfiles/")

    # Filter for StormEvents_details files from 1999-2025
    pattern = r'StormEvents_details-ftp_v1\.0_d(\d{4})_c.*\.csv\.gz'
    selected_files = []
        
    for file in files:
        match = re.match(pattern, file)
        if match:
            year = int(match.group(1))
            if 1999 <= year <= 2024:
                selected_files.append(file)


    # Use the selected_files list
    print(f"Selected {len(selected_files)} StormEvents_details files:")
    for i, filename in enumerate(selected_files, 1):
        year = re.search(r"d(\d{4})", filename).group(1)
        print(f"{filename}")


    table_exists = dbt.query("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'noaa_stormevents_ftp_all'
        );
    """).iloc[0, 0]

    # get all files identified in 'filenames' and populate a df for cleaning
    if table_exists:
        print("Loading existing data from database...")
        df_all_storms = dbt.query("SELECT * FROM noaa_stormevents_ftp_all")
        print(f"Loaded {len(df_all_storms):,} rows from database")
    else:
        print("Table doesn't exist. Downloading and processing FTP data...")

        base_url = "ftp://ftp.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
        all_storm_data = []

        print(f"Processing {len(selected_files)} Storm Events files...")

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
            print("Saving data to database...")
            dbt.load_data(
                df_all_storms, "noaa_stormevents_ftp_all", if_exists="replace"
            )
            print("Data saved to database.")
        else:
            print("No data was successfully loaded")

    # Drop unneeded columns to reduce memory usage
    # Keeping injuries and deaths for severity analysis
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

    # combine BEGIN_YEARMONTH and BEGIN_DAY into a single DATE column and convert to datetime, drop original columns
    # create YEAR column for filtering later

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
    df_all_storms_comb["YEAR"] = df_all_storms_comb["DATE"].dt.year

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

    # clean FIPS due to historical changes and non populated areas (marine, unincorporated, etc)

    df_clean = df_all_storms_comb.copy()
    df_clean = df_clean[
        (df_clean["CO_FIPS"] >= "01001")
        & (df_clean["CO_FIPS"] <= "56045")
        & (~df_clean["CO_FIPS"].str.startswith("99"))
    ].copy()

    # Filter data to only include events with direct and indirect deaths or injuries
    severe_events = df_clean[
        (df_clean["DEATHS_DIRECT"] > 0)
        | (df_clean["INJURIES_DIRECT"] > 0)
        | (df_clean["DEATHS_INDIRECT"] > 0)
        | (df_clean["INJURIES_INDIRECT"] > 0)
    ].copy()

    # Group by episode and county fips to get unique events
    county_episodes = (
        severe_events.groupby(["CO_FIPS", "EPISODE_ID", "YEAR"])
        .agg(
            {
                "DEATHS_DIRECT": "sum",
                "DEATHS_INDIRECT": "sum",
                "INJURIES_INDIRECT": "sum",
                "INJURIES_DIRECT": "sum",
                "EVENT_TYPE": lambda x: ", ".join(sorted(set(x))),
                "DATE": "first",
            }
        )
        .reset_index()
    )

    # Aggregate by county-year with both count and other metrics
    annual_episodes = (
        county_episodes.groupby(["CO_FIPS", "YEAR"])
        .agg(
            {
                "EPISODE_ID": "count",  # This gives us the event_count
                "DEATHS_DIRECT": "sum",
                "DEATHS_INDIRECT": "sum",
                "INJURIES_DIRECT": "sum",
                "INJURIES_INDIRECT": "sum",
                "EVENT_TYPE": lambda x: ", ".join(sorted(set(x))),
                "DATE": "first",
            }
        )
        .reset_index()
    )

    # Rename columns to match your existing structure
    annual_episodes.columns = [
        "county_fips",
        "year",
        "event_count",
        "total_deaths_direct",
        "total_deaths_indirect",
        "total_injuries_direct",
        "total_injuries_indirect",
        "event_types",
        "first_event_date",
    ]

    # Create complete county-year combinations for counties that had at least one severe event
    all_counties = annual_episodes["county_fips"].unique()
    all_years = range(annual_episodes["year"].min(), annual_episodes["year"].max() + 1)

    # Create all combinations
    complete_combinations = pd.MultiIndex.from_product(
        [all_counties, all_years], names=["county_fips", "year"]
    ).to_frame(index=False)

    # Merge and fill missing with 0 for numeric columns, empty string for text columns
    annual_episodes = complete_combinations.merge(
        annual_episodes, on=["county_fips", "year"], how="left"
    ).fillna(
        {
            "event_count": 0,
            "total_deaths_direct": 0,
            "total_deaths_indirect": 0,
            "total_injuries_direct": 0,
            "total_injuries_indirect": 0,
            "event_types": "",
            "first_event_date": pd.NaT,
        }
    )

    annual_episodes[
        [
            "event_count",
            "total_deaths_direct",
            "total_deaths_indirect",
            "total_injuries_direct",
            "total_injuries_indirect",
        ]
    ] = annual_episodes[
        [
            "event_count",
            "total_deaths_direct",
            "total_deaths_indirect",
            "total_injuries_direct",
            "total_injuries_indirect",
        ]
    ].astype(int)

    # Load cleaned data into the database
    dbt.load_data(annual_episodes, "NOAA_STORM_EPISODES", if_exists="replace")
    dbt.load_data(df_clean, "NOAA_STORM_EVENTS", if_exists="replace")
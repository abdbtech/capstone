
# import libraries
# NOTE: g"""Compound Poisson Model
Poisson, rate, component

"""

dbt.print_section_start(
    "Poisson Risk Modeling", "Calculating county-level disaster frequency parameters"
)rs should be edited to include local paths and credentials before use.
# I"""Compound Poisson Model
Weibull, vulnerability, component
"""
dbt.print_section_start(
    "Weibull Severity Modeling",
    "Fitting severity distribution using vulnerability and casualty data",
)l_vars.py is created in the root dir remove the ignore/ prefix in the import statement below.
import ignore.global_vars as gv"""Compound Poisson Model
Model integration

"""
dbt.print_section_start(
    "Risk Simulation", "Running Monte Carlo simulations for all counties"
) db_tools as dbt
import zipfile
import os"""QGIS coordinates
"""
dbt.print_section_start(
    "QGIS Spatial Export", "Creating spatial tables for QGIS visualization"
)ort pandas as pd
import geopandas as gpd
import re
import tqdm
import numpy as np
from scipy.stats import weibull_m"""K-Means Clustering

"""
dbt.print_section_start(
    "Depot Location Analysis",
    f"K-Means clustering for {n_depots} strategic depot locations",
)m sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from scipy.stats import weibull_min
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime


# NOTE: Set to True if a full rebuild is required, set to False to skip table builds. Search 'REBUILD' to see which sections are effected.
REBUILD = True
# change n_depots to change number of clusters
n_depots = 36
# change n_depots to change number of clusters
n_depots = 36

"""
Compound Poisson Model
Poisson, rate, component

"""

print_section_start(
    "Poisson Risk Modeling", "Calculating county-level disaster frequency parameters"
)

# Load data for poisson analysis
dbt.print_status("Loading data for Poisson analysis...")
geography_columns = dbt.query("SELECT * FROM geography_columns")
df_noaa = dbt.query('SELECT * FROM "NOAA_STORM_EPISODES" ORDER BY county_fips, year')
dbt.print_status(f"Loaded {len(df_noaa)} county-year storm episode records")

# get summary statistics for each county. Group by county_fips
county_lambdas = (
    df_noaa.groupby("county_fips")
    .agg(
        {"event_count": ["mean", "var", "std", "count", "sum"], "year": ["min", "max"]}
    )
    .round(4)
)

# Flatten column names
county_lambdas.columns = ["_".join(col).strip() for col in county_lambdas.columns]
county_lambdas = county_lambdas.reset_index()

# Rename for clarity
county_lambdas.rename(
    columns={
        "event_count_mean": "lambda_hat",
        "event_count_var": "variance",
        "event_count_std": "std_dev",
        "event_count_count": "years_observed",
        "event_count_sum": "total_events",
        "year_min": "first_year",
        "year_max": "last_year",
    },
    inplace=True,
)

# Calculate overdispersion ratio (variance/mean)
county_lambdas["overdispersion_ratio"] = np.where(
    county_lambdas["lambda_hat"] > 0,
    county_lambdas["variance"] / county_lambdas["lambda_hat"],
    np.nan,
)

# make a dataframe of the poisson risk parameters
poisson_risk_params = county_lambdas[
    ["county_fips", "lambda_hat", "years_observed", "total_events"]
].copy()

# Add confidence intervals for lambda estimates NOTE: not yet implemented, generating these in case we need them
poisson_risk_params["lambda_se"] = np.sqrt(
    poisson_risk_params["lambda_hat"] / poisson_risk_params["years_observed"]
)
poisson_risk_params["lambda_ci_lower"] = (
    poisson_risk_params["lambda_hat"] - 1.96 * poisson_risk_params["lambda_se"]
)
poisson_risk_params["lambda_ci_upper"] = (
    poisson_risk_params["lambda_hat"] + 1.96 * poisson_risk_params["lambda_se"]
)

# set any negative lower CI bounds to zero
poisson_risk_params["lambda_ci_lower"] = np.maximum(
    0, poisson_risk_params["lambda_ci_lower"]
)

# calculate the probability of at least one event in a year
# P(≥1 disaster) = 1 - e^(-λ)
poisson_risk_params["prob_at_least_one_event"] = 1 - np.exp(
    -poisson_risk_params["lambda_hat"]
)


dbt.print_section_complete(
    "Poisson Risk Modeling",
    f"Calculated frequency parameters for {len(poisson_risk_params)} counties",
)

"""
Compound Poisson Model
Weibull, vulnerability, component
"""
print_section_start(
    "Weibull Severity Modeling",
    "Fitting severity distribution using vulnerability and casualty data",
)

# Load tables from database
dbt.print_status("Loading storm and census data for severity modeling...")
df_noaa_episodes = dbt.query('SELECT * FROM "NOAA_STORM_EPISODES"')
df_noaa_events = dbt.query('SELECT * FROM "NOAA_STORM_EVENTS"')
df_census = dbt.query('SELECT * FROM "census_resilience"')

# Generate casualty rate and merge noaa_episodes with census on county_fips
df_noaa_episodes["casualties"] = (
    df_noaa_episodes["total_injuries_direct"]
    + df_noaa_episodes["total_deaths_direct"]
    + df_noaa_episodes["total_injuries_indirect"]
    + df_noaa_episodes["total_deaths_indirect"]
)
noaa_episodes_census_merge = df_noaa_episodes.merge(
    df_census, left_on="county_fips", right_on="County_fips"
)
noaa_episodes_census_merge["casualty_rate"] = (
    noaa_episodes_census_merge["casualties"]
    / noaa_episodes_census_merge["POPUNI"]
    * 1000
)
noaa_census = noaa_episodes_census_merge.copy()
noaa_census[noaa_census["casualties"] > 1].describe()

# combine vunerabiliyt predictors with simple addition and create intensity rate.
# Multiplicative was chose for amplification effect of casualties on a vunerable population
# Intensity = casualty rate (per 1000) * 1 + (vunerability_rate /100)
noaa_census["vunerability_rate"] = noaa_census["PRED12_PE"] + noaa_census["PRED3_PE"]
noaa_census["intensity"] = noaa_census["casualty_rate"] * (
    1 + (noaa_census["vunerability_rate"] / 100)
)

# Calculate state averages for vulnerability imputation
df_census["state_fips"] = df_census["County_fips"].str[:2]
state_vulnerability_avg = df_census.groupby("state_fips")[
    ["PRED12_PE", "PRED3_PE"]
].mean()

# Perform left join to keep all NOAA data. strip state FIPS for averaging later
noaa_census_full = df_noaa_episodes.merge(
    df_census, left_on="county_fips", right_on="County_fips", how="left"
)
noaa_census_full["state_fips"] = noaa_census_full["county_fips"].str[:2]
noaa_census_full.shape

# Impute missing vulnerability data using state averages
state_avg_mapping = state_vulnerability_avg.to_dict()


def impute_vulnerability(row, vulnerability_col, state_col="state_fips"):
    if pd.isna(row[vulnerability_col]):
        state = row[state_col]
        if state in state_avg_mapping[vulnerability_col]:
            return state_avg_mapping[vulnerability_col][state]
        else:
            # If state not in mapping, use overall average
            return df_census[vulnerability_col].mean()
    return row[vulnerability_col]


# Apply imputation
dbt.print_status("Imputing missing vulnerability data using state averages...")
noaa_census_full["PRED12_PE_imputed"] = [
    impute_vulnerability(row, "PRED12_PE")
    for _, row in tqdm.tqdm(
        noaa_census_full.iterrows(),
        total=len(noaa_census_full),
        desc="Imputing PRED12_PE",
    )
]
noaa_census_full["PRED3_PE_imputed"] = [
    impute_vulnerability(row, "PRED3_PE")
    for _, row in tqdm.tqdm(
        noaa_census_full.iterrows(),
        total=len(noaa_census_full),
        desc="Imputing PRED3_PE",
    )
]

# Recalculate intensity with imputed vulnerability data
# Use POPUNI from census data, impute if missing using national average
national_avg_pop = df_census["POPUNI"].mean()
noaa_census_full["POPUNI_imputed"] = noaa_census_full["POPUNI"].fillna(national_avg_pop)

# Calculate casualties, casualty rate, vulnerability rate, and intensity
noaa_census_full["casualties"] = (
    noaa_census_full["total_injuries_direct"]
    + noaa_census_full["total_deaths_direct"]
    + noaa_census_full["total_injuries_indirect"]
    + noaa_census_full["total_deaths_indirect"]
)

noaa_census_full["casualty_rate"] = (
    noaa_census_full["casualties"] / noaa_census_full["POPUNI_imputed"] * 1000
)
noaa_census_full["vulnerability_rate"] = (
    noaa_census_full["PRED12_PE_imputed"] + noaa_census_full["PRED3_PE_imputed"]
)
noaa_census_full["intensity"] = noaa_census_full["casualty_rate"] * (
    1 + (noaa_census_full["vulnerability_rate"] / 100)
)

# Filter for events with intensity > 0 for distribution fitting
intensity_data = noaa_census_full[noaa_census_full["intensity"] > 0]["intensity"]

# Fit distribution to Wiebull using .fit MLE
weibull_params = weibull_min.fit(intensity_data, floc=0)
weibull_shape, weibull_loc, weibull_scale = weibull_params

# Used scipy weibull_min for modeling "min extreme value" of derived "intensity"
# shape: k parameter describing skewedness
# scale: lambda parameter for spread
# location: shift parameter, locked at 0 for this implementation.
severity_distribution = {
    "distribution": "weibull_min",
    "shape": weibull_shape,
    "scale": weibull_scale,
    "location": weibull_loc,
}

dbt.print_section_complete(
    "Weibull Severity Modeling",
    f"Fitted Weibull distribution (shape={weibull_shape:.3f}, scale={weibull_scale:.3f})",
)

"""
Compound Poisson Model
Model integration

"""
print_section_start(
    "Risk Simulation", "Running Monte Carlo simulations for all counties"
)


def simulate_compound_poisson_risk(county_fips, n_simulations=10000):
    # Get county's λ (frequency)
    lambda_i = poisson_risk_params[poisson_risk_params["county_fips"] == county_fips][
        "lambda_hat"
    ].iloc[0]

    # Simulate total risk for each year
    total_risks = []
    for _ in range(n_simulations):
        # Step 1: Generate number of events N ~ Poisson(λ)
        n_events = np.random.poisson(lambda_i)

        # Step 2: Generate severity for each event Y ~ Weibull
        if n_events > 0:
            severities = weibull_min.rvs(
                weibull_shape, loc=weibull_loc, scale=weibull_scale, size=n_events
            )
            total_risk = np.sum(severities)
        else:
            total_risk = 0

        total_risks.append(total_risk)

    return np.array(total_risks)


# Apply to all counties
dbt.print_status(f"Starting risk simulations for {len(poisson_risk_params)} counties...")
county_risks = {}
for county in tqdm.tqdm(
    poisson_risk_params["county_fips"], desc="Simulating county risks"
):
    county_risks[county] = simulate_compound_poisson_risk(county)

dbt.print_section_complete(
    "Risk Simulation",
    f"Completed Monte Carlo simulations for {len(county_risks)} counties",
)


"""
QGIS coordinates
"""
print_section_start(
    "QGIS Spatial Export", "Creating spatial tables for QGIS visualization"
)

# Create spatial table with correct coordinates for QGIS mapping
if REBUILD:
    # Define the columns we want to include in the spatial data
    risk_columns = [
        "county_fips",
        "lambda_hat",
        "total_events",
        "years_observed",
        "lambda_ci_lower",
        "lambda_ci_upper",
        "prob_at_least_one_event",
    ]

    try:
        dbt.print_status("Preparing spatial data for QGIS...")
        # Use the reusable function to prepare spatial data
        lambda_map_clean, db_data = dbt.prepare_spatial_data_for_qgis(
            dataframe=poisson_risk_params,
            geometry_source="nri_shapefile",
            data_columns=risk_columns,
            county_fips_col="county_fips",
            impute_missing=True,
            add_rounded_cols=True,
        )

        # Create the corrected spatial table directly with proper coordinates
        print("Creating spatial table with corrected coordinates...")

        # Save intermediate data
        dbt.load_data(db_data, "disaster_risk_spatial_temp", if_exists="replace")

        # Create final spatial table with proper coordinate system
        spatial_table_sql = """
        DROP TABLE IF EXISTS disaster_risk_counties_spatial_corrected;
        
        CREATE TABLE disaster_risk_counties_spatial_corrected AS
        SELECT 
            county_fips,
            lambda_hat,
            lambda_rounded,
            prob_at_least_one_event,
            total_events,
            years_observed,
            lambda_ci_lower,
            lambda_ci_upper,
            ST_GeomFromText(geometry_wkt, 4326) as geometry
        FROM disaster_risk_spatial_temp;

        CREATE INDEX idx_disaster_risk_counties_spatial_corrected_geom 
        ON disaster_risk_counties_spatial_corrected USING GIST (geometry);

        ALTER TABLE disaster_risk_counties_spatial_corrected 
        ADD CONSTRAINT pk_disaster_risk_counties_spatial_corrected PRIMARY KEY (county_fips);
        
        -- Clean up temporary table
        DROP TABLE IF EXISTS disaster_risk_spatial_temp;
        """

        dbt.execute_sql(spatial_table_sql)
        print("✓ Spatial table created: disaster_risk_counties_spatial_corrected")

    except Exception as e:
        print(f"Error creating spatial data: {e}")
        print("No geographic data available")

else:
    dbt.print_status("Skip rebuilding spatial table", "SKIP")

dbt.print_section_complete("QGIS Spatial Export", "Spatial data preparation completed")

county_geo_risk_query = """
SELECT 
    s.county_fips,
    ST_Y(ST_Centroid(s.geometry)) as latitude,
    ST_X(ST_Centroid(s.geometry)) as longitude,
    c.expected_annual_loss,
    c.var_95,
    s.lambda_hat
FROM disaster_risk_counties_spatial_corrected s
JOIN disaster_risk_clusters c ON s.county_fips = c.county_fips
WHERE s.geometry IS NOT NULL
ORDER BY s.county_fips
"""

# load counties including geographic and risk data
try:
    county_data = dbt.query(county_geo_risk_query)
    print(f"Loaded {len(county_data)} counties with geographic and risk data")

    if len(county_data) > 0:
        # Data validation and cleaning
        print("Data validation:")
        print(
            f"Latitude range: {county_data['latitude'].min():.3f} to {county_data['latitude'].max():.3f}"
        )
        print(
            f"Longitude range: {county_data['longitude'].min():.3f} to {county_data['longitude'].max():.3f}"
        )
        print(f"NULL values: {county_data.isnull().sum().sum()}")
    else:
        print("No geographic data available - creating empty DataFrame")
        county_data = pd.DataFrame()
except Exception as e:
    print(f"Error loading geographic data: {e}")
    county_data = pd.DataFrame()

# Filter out invalid coordinates
valid_coords = (
    (county_data["latitude"] >= -90)
    & (county_data["latitude"] <= 90)
    & (county_data["longitude"] >= -180)
    & (county_data["longitude"] <= 180)
    & county_data["latitude"].notna()
    & county_data["longitude"].notna()
)

county_data_clean = county_data[valid_coords].copy()

"""
K-Means Clustering

"""
print_section_start(
    "Depot Location Analysis",
    f"K-Means clustering for {n_depots} strategic depot locations",
)

dbt.print_status(f"Preparing geographic data for {len(county_data_clean)} counties...")
# Geographic clustering
geo_features = county_data_clean[["latitude", "longitude"]].values
geo_scaler = StandardScaler()
geo_features_scaled = geo_scaler.fit_transform(geo_features)
geo_kmeans = KMeans(n_clusters=n_depots, random_state=36, n_init=10)
county_data_clean["depot_service_area"] = geo_kmeans.fit_predict(geo_features_scaled)

# Step 3: Calculate depot locations (centroids of service areas)
dbt.print_status("Calculating optimal depot locations...")
depot_locations = []
for depot_id in tqdm.tqdm(range(n_depots), desc="Computing depot locations"):
    service_counties = county_data_clean[
        county_data_clean["depot_service_area"] == depot_id
    ]

    # Weighted centroid based on risk levels
    total_risk = service_counties["expected_annual_loss"].sum()
    if total_risk > 0:
        weighted_lat = (
            service_counties["latitude"] * service_counties["expected_annual_loss"]
        ).sum() / total_risk
        weighted_lon = (
            service_counties["longitude"] * service_counties["expected_annual_loss"]
        ).sum() / total_risk
    else:
        weighted_lat = service_counties["latitude"].mean()
        weighted_lon = service_counties["longitude"].mean()

    depot_locations.append(
        {
            "depot_id": depot_id,
            "latitude": weighted_lat,
            "longitude": weighted_lon,
            "counties_served": len(service_counties),
            "total_risk_served": service_counties["expected_annual_loss"].sum(),
            "avg_risk_per_county": service_counties["expected_annual_loss"].mean(),
        }
    )

# Convert depot locations to DataFrame
depot_df = pd.DataFrame(depot_locations)

# Create county assignments DataFrame
county_assignments = county_data_clean[["county_fips", "depot_service_area"]].copy()
county_assignments = county_assignments.rename(
    columns={"depot_service_area": "depot_id"}
)

# Calculate service statistics for each county
dbt.print_status("Calculating service statistics and distances...")
service_area_stats = []
for _, county in tqdm.tqdm(
    county_data_clean.iterrows(),
    total=len(county_data_clean),
    desc="Calculating distances",
):
    depot_id = county["depot_service_area"]
    depot_info = depot_df[depot_df["depot_id"] == depot_id].iloc[0]

    # Calculate distance to depot (Haversine formula)
    from math import radians, cos, sin, asin, sqrt

    def haversine(lon1, lat1, lon2, lat2):
        """Calculate the great circle distance in kilometers between two points on the earth"""
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r

    distance = haversine(
        county["longitude"],
        county["latitude"],
        depot_info["longitude"],
        depot_info["latitude"],
    )

    service_area_stats.append(
        {
            "county_fips": county["county_fips"],
            "depot_id": depot_id,
            "distance_to_depot_km": distance,
            "county_risk": county["expected_annual_loss"],
        }
    )

service_stats_df = pd.DataFrame(service_area_stats)

# Load depot-related tables to database for QGIS and downstream use
print("Loading depot analysis tables to database...")
dbt.load_data(depot_df, "strategic_depot_locations", if_exists="replace")
print("Data loaded successfully into strategic_depot_locations")

dbt.load_data(county_assignments, "county_depot_assignments", if_exists="replace")
print("Data loaded successfully into county_depot_assignments")

dbt.load_data(service_stats_df, "depot_service_statistics", if_exists="replace")
print("Data loaded successfully into depot_service_statistics")

dbt.print_section_complete(
    "Depot Location Analysis",
    f"Created {n_depots} depot locations with service assignments",
)

# print tables names for current db specified in gv
dbt.print_status("Retrieving final database table list...")
table_names = dbt.query(
    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
)
print("\nDatabase tables created:")
for table in table_names["table_name"]:
    print(f"  ✓ {table}")

dbt.print_section_start(
    "ETL PIPELINE COMPLETE", "All data processing and analysis completed successfully"
)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"[{timestamp}] ETL pipeline execution finished.")
print(f"{'=' * 60}")
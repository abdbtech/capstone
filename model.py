# import libraries
# NOTE: global_vars should be edited to include local paths and credentials before use.
import math
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
import db_tools as dbt

# NOTE: Model variables
n_depots = 36  # Set number of depots for clustering analysis

if __name__ == "__main__":

    '''
    NRI Shapefile ETL
    '''
    print("Starting modeling process...")

    # Load the NOAA data into a DataFrame for analysis
    df_noaa = dbt.query("SELECT * FROM \"NOAA_STORM_EPISODES\" ORDER BY county_fips, year")

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

    # Define the columns
    risk_columns = [
        "county_fips",
        "lambda_hat",
        "total_events",
        "years_observed",
        "lambda_ci_lower",
        "lambda_ci_upper",
        "prob_at_least_one_event"
    ]
    
    try:
        # Use the reusable function to prepare spatial data
        lambda_map_clean, db_data = dbt.prepare_spatial_data_for_qgis(
            dataframe=poisson_risk_params,
            geometry_source='nri_shapefile',
            data_columns=risk_columns,
            county_fips_col='county_fips',
            impute_missing=True,
            add_rounded_cols=True
        )
        
        # Save to database
        print("Saving spatial data to database...")
        dbt.load_data(db_data, "disaster_risk_spatial", if_exists="replace")
        
        # Create spatial table with PostGIS geometry
        print("Creating PostGIS spatial table...")
        spatial_table_sql = """
        DROP TABLE IF EXISTS disaster_risk_counties_spatial;

        CREATE TABLE disaster_risk_counties_spatial AS
        SELECT 
            county_fips,
            lambda_hat_rounded,
            prob_at_least_one_event,
            total_events,
            years_observed,
            lambda_ci_lower,
            lambda_ci_upper,
            ST_GeomFromText(geometry_wkt, 4326) as geometry
        FROM disaster_risk_spatial;

        -- Add spatial index for performance
        CREATE INDEX idx_disaster_risk_counties_spatial_geom 
        ON disaster_risk_counties_spatial USING GIST (geometry);

        -- Add primary key
        ALTER TABLE disaster_risk_counties_spatial 
        ADD CONSTRAINT pk_disaster_risk_counties_spatial PRIMARY KEY (county_fips);
        """
        
        dbt.execute_sql(spatial_table_sql)
        print("Spatial database table created: disaster_risk_counties_spatial")

    except Exception as e:
        print(f"Error creating spatial data: {e}")
        print("No geographic data available")

        # create table for disaster counts by county

    event_types_sql = """
    SELECT DISTINCT "EVENT_TYPE" 
    FROM "NOAA_STORM_EVENTS" 
    WHERE ("INJURIES_DIRECT" > 0 OR "DEATHS_DIRECT" > 0)
    ORDER BY "EVENT_TYPE";
    """

    event_types = dbt.query(event_types_sql)
    # Build dynamic pivot columns
    pivot_columns = []
    for event_type in event_types["EVENT_TYPE"]:
        safe_name = (
            event_type.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        pivot_columns.append(
            f"SUM(CASE WHEN \"EVENT_TYPE\" = '{event_type}' THEN event_count ELSE 0 END) as {safe_name}_severe"
        )

    pivot_sql = f"""
    WITH severe_events AS (
        SELECT 
            "CO_FIPS" as county_fips,
            "EVENT_TYPE",
            COUNT(*) as event_count
        FROM "NOAA_STORM_EVENTS" 
        WHERE ("INJURIES_DIRECT" > 0 OR "DEATHS_DIRECT" > 0)
        AND "CO_FIPS" IS NOT NULL
        GROUP BY "CO_FIPS", "EVENT_TYPE"
    ),
    pivoted_events AS (
        SELECT 
            county_fips,
            {",".join(pivot_columns)},
            COUNT(DISTINCT "EVENT_TYPE") as severe_event_types,
            SUM(event_count) as total_severe_events
        FROM severe_events
        GROUP BY county_fips
    )
    SELECT 
        s.*,
        p.{",p.".join([col.split(" as ")[1] for col in pivot_columns])},
        p.severe_event_types,
        p.total_severe_events
    FROM disaster_risk_counties_spatial s
    LEFT JOIN pivoted_events p ON s.county_fips = p.county_fips
    ORDER BY s.county_fips;
    """

    pivot_result = dbt.query(pivot_sql)

    # Create table with event counts per county
    # This table is used for troubleshooting and future functionality
    # or for county level historical disaster exploration

    create_enhanced_table_sql = f"""
    DROP TABLE IF EXISTS disaster_risk_counties_event_counts;

    CREATE TABLE disaster_risk_counties_event_counts AS
    {pivot_sql};

    -- Add indexes
    CREATE INDEX idx_disaster_risk_event_counts_geom ON disaster_risk_counties_event_counts USING GIST (geometry);
    ALTER TABLE disaster_risk_counties_event_counts ADD CONSTRAINT pk_disaster_risk_event_counts PRIMARY KEY (county_fips);
    """

    dbt.execute_sql(create_enhanced_table_sql)
    print("Enhanced table created: disaster_risk_counties_event_counts")


    '''
    Weibull Severity component
    '''

    # Load tables from database
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

    # Impute missing vulnerability data using state averages
    # First, create a mapping of state averages
    state_avg_mapping = state_vulnerability_avg.to_dict()


    # Function to impute missing values
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
    noaa_census_full["PRED12_PE_imputed"] = noaa_census_full.apply(
        lambda row: impute_vulnerability(row, "PRED12_PE"), axis=1
    )
    noaa_census_full["PRED3_PE_imputed"] = noaa_census_full.apply(
        lambda row: impute_vulnerability(row, "PRED3_PE"), axis=1
    )

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

    '''
    Compound Model Integration
    '''

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
    
    # Apply to all counties in your dataset
    county_risks = {}
    for county in tqdm.tqdm(poisson_risk_params["county_fips"]):
        county_risks[county] = simulate_compound_poisson_risk(county)

    # Create disaster_risk_clusters table from county_risks simulation results
    print("Creating disaster_risk_clusters table from simulation results...")

    # Convert county_risks dictionary to DataFrame
    risk_data = []
    for county_fips, risk_array in county_risks.items():
        risk_data.append(
            {
                "county_fips": county_fips,
                "expected_annual_loss": np.mean(risk_array),
                "var_95": np.percentile(risk_array, 95),
                "var_99": np.percentile(risk_array, 99),
                "std_dev": np.std(risk_array),
            }
        )

    county_risk_df = pd.DataFrame(risk_data)

    # Save to database
    dbt.load_data(county_risk_df, "disaster_risk_clusters", if_exists="replace")
    print(f"Created disaster_risk_clusters table with {len(county_risk_df)} counties")

    # coordinates are EPSG:3857 but stored with wrong SRID
    # transforming to WGS84
    fix_coordinates_sql = """
    DROP TABLE IF EXISTS disaster_risk_counties_spatial_corrected;
    CREATE TABLE disaster_risk_counties_spatial_corrected AS
    SELECT 
        county_fips,
        lambda_hat_rounded,
        prob_at_least_one_event,
        total_events,
        years_observed,
        lambda_ci_lower,
        lambda_ci_upper,
        ST_Transform(ST_SetSRID(geometry, 3857), 4326) as geometry
    FROM disaster_risk_counties_spatial;

    CREATE INDEX idx_disaster_risk_counties_spatial_corrected_geom 
    ON disaster_risk_counties_spatial_corrected USING GIST (geometry);

    ALTER TABLE disaster_risk_counties_spatial_corrected 
    ADD CONSTRAINT pk_disaster_risk_counties_spatial_corrected PRIMARY KEY (county_fips);
    """

    # extract county centroids and risk data with correctly formatted coordinates
    dbt.execute_sql(fix_coordinates_sql)

    county_geo_risk_query = """
    SELECT 
        s.county_fips,
        ST_Y(ST_Centroid(s.geometry)) as latitude,
        ST_X(ST_Centroid(s.geometry)) as longitude,
        c.expected_annual_loss,
        c.var_95,
        s.lambda_hat_rounded as lambda_hat
    FROM disaster_risk_counties_spatial_corrected s
    JOIN disaster_risk_clusters c ON s.county_fips = c.county_fips
    WHERE s.geometry IS NOT NULL
    ORDER BY s.county_fips
    """

    # load counties including geographic and risk data
    county_data = dbt.query(county_geo_risk_query)
    print(f"Loaded {len(county_data)} counties with geographic and risk data")

    # Data validation and cleaning
    print("Data validation:")
    print(
        f"Latitude range: {county_data['latitude'].min():.3f} to {county_data['latitude'].max():.3f}"
    )
    print(
        f"Longitude range: {county_data['longitude'].min():.3f} to {county_data['longitude'].max():.3f}"
    )
    print(f"NULL values: {county_data.isnull().sum().sum()}")

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
    print(f"After cleaning: {len(county_data_clean)} counties with valid coordinates")

    # Geographic clustering for depot placement (pure geography)
    geo_features = county_data_clean[["latitude", "longitude"]].values
    geo_scaler = StandardScaler()
    geo_features_scaled = geo_scaler.fit_transform(geo_features)

    # set clusters are run, set at top with n_depots = 

    geo_kmeans = KMeans(n_clusters=n_depots, random_state=36, n_init=10)
    county_data_clean["depot_service_area"] = geo_kmeans.fit_predict(geo_features_scaled)

    # Step 3: Calculate depot locations (centroids of service areas)
    depot_locations = []
    for depot_id in range(n_depots):
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

    depot_df = pd.DataFrame(depot_locations)

    # Calculate service area statistics with error handling
    def calculate_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points using geodesic distance with validation"""
        try:
            # Validate coordinates before calculation
            if any(pd.isna([lat1, lon1, lat2, lon2])):
                return np.nan
            if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
                return np.nan
            if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
                return np.nan

            return geodesic((lat1, lon1), (lat2, lon2)).kilometers
        except Exception as e:
            print(f"Distance calculation error: {e}")
            return np.nan


    service_area_stats = []
    for _, county in county_data_clean.iterrows():
        depot_id = county["depot_service_area"]
        depot_info = depot_df[depot_df["depot_id"] == depot_id].iloc[0]

        distance = calculate_distance(
            county["latitude"],
            county["longitude"],
            depot_info["latitude"],
            depot_info["longitude"],
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

    # Remove rows with invalid distances
    service_stats_df = service_stats_df.dropna(subset=["distance_to_depot_km"])
    print(f"Valid distance calculations: {len(service_stats_df)}")


    # Analyze depot performance
    depot_performance = (
        service_stats_df.groupby("depot_id")
        .agg(
            {
                "distance_to_depot_km": ["mean", "max", "std"],
                "county_risk": ["sum", "mean", "count"],
                "county_fips": "count",
            }
        )
        .round(2)
    )

    depot_performance.columns = ["_".join(col).strip() for col in depot_performance.columns]
    depot_performance = depot_performance.reset_index()

    print("\nDepot Performance Analysis:")
    print(depot_performance)

    # Step 6: Save depot results to database
    print("\nSaving depot placement results to database...")

    # Save depot locations
    dbt.load_data(depot_df, "strategic_depot_locations", if_exists="replace")

    # Save county assignments
    county_assignments = county_data_clean[["county_fips", "depot_service_area"]].copy()
    county_assignments.rename(columns={"depot_service_area": "depot_id"}, inplace=True)
    dbt.load_data(county_assignments, "county_depot_assignments", if_exists="replace")

    # Save detailed service statistics
    dbt.load_data(service_stats_df, "depot_service_statistics", if_exists="replace")

    # Create spatial table for QGIS mapping
    create_depot_spatial_sql = """
    -- Create depot points table for QGIS
    DROP TABLE IF EXISTS strategic_depot_points;

    CREATE TABLE strategic_depot_points AS
    SELECT 
        depot_id,
        latitude,
        longitude,
        counties_served,
        total_risk_served,
        avg_risk_per_county,
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326) as geometry
    FROM strategic_depot_locations;

    -- Add spatial index
    CREATE INDEX idx_strategic_depot_points_geom 
    ON strategic_depot_points USING GIST (geometry);

    -- Create county service areas table for QGIS
    DROP TABLE IF EXISTS county_service_areas_spatial;

    CREATE TABLE county_service_areas_spatial AS
    SELECT 
        c.county_fips,
        c.depot_id,
        s.distance_to_depot_km,
        s.county_risk,
        g.geometry
    FROM county_depot_assignments c
    JOIN depot_service_statistics s ON c.county_fips = s.county_fips
    JOIN disaster_risk_counties_spatial_corrected g ON c.county_fips = g.county_fips;

    -- Add spatial index
    CREATE INDEX idx_county_service_areas_spatial_geom 
    ON county_service_areas_spatial USING GIST (geometry);
    """

    dbt.execute_sql(create_depot_spatial_sql)

    '''
    Create compound_poisson_map table for comprehensive risk metrics
    '''

    # Calculate expected severity from Weibull distribution
    expected_severity = weibull_scale * math.gamma(1 + 1/weibull_shape)
    print(f"Expected Weibull severity value: {expected_severity:.4f}")

    # Create comprehensive risk metrics table
    compound_poisson_map_sql = """
    -- Create table combining all risk components for spatial mapping
    DROP TABLE IF EXISTS compound_poisson_map;

    CREATE TABLE compound_poisson_map AS
    SELECT 
        s.county_fips,
        s.geometry,
        -- Poisson frequency component
        p.lambda_hat_rounded as poisson_frequency,
        p.prob_at_least_one_event,
        p.total_events,
        p.years_observed,
        -- Compound risk metrics
        c.expected_annual_loss as compound_risk,
        c.var_95 as risk_95th_percentile,
        c.var_99 as risk_99th_percentile,
        c.std_dev as risk_std_dev,
        -- Severity component (constant across all counties from fitted distribution)
        {} as expected_severity,
        -- Risk ratios and rankings
        PERCENT_RANK() OVER (ORDER BY c.expected_annual_loss) as risk_percentile_rank,
        NTILE(10) OVER (ORDER BY c.expected_annual_loss) as risk_decile,
        CASE 
            WHEN c.expected_annual_loss >= (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY expected_annual_loss) FROM disaster_risk_clusters) THEN 'Very High'
            WHEN c.expected_annual_loss >= (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY expected_annual_loss) FROM disaster_risk_clusters) THEN 'High'
            WHEN c.expected_annual_loss >= (SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY expected_annual_loss) FROM disaster_risk_clusters) THEN 'Medium'
            WHEN c.expected_annual_loss >= (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY expected_annual_loss) FROM disaster_risk_clusters) THEN 'Low'
            ELSE 'Very Low'
        END as risk_category
    FROM disaster_risk_counties_spatial_corrected s
    JOIN disaster_risk_clusters c ON s.county_fips = c.county_fips
    LEFT JOIN (
        SELECT 
            county_fips,
            lambda_hat_rounded,
            prob_at_least_one_event,
            total_events,
            years_observed
        FROM disaster_risk_counties_spatial_corrected
    ) p ON s.county_fips = p.county_fips
    WHERE s.geometry IS NOT NULL;

    -- Add spatial index
    CREATE INDEX idx_compound_poisson_map_geom 
    ON compound_poisson_map USING GIST (geometry);

    -- Add primary key
    ALTER TABLE compound_poisson_map 
    ADD CONSTRAINT pk_compound_poisson_map PRIMARY KEY (county_fips);
    """.format(expected_severity)

    # Execute the SQL
    dbt.execute_sql(compound_poisson_map_sql)

    print("✓ Created compound_poisson_map table with:")
    print("  - Poisson frequency (lambda_hat)")
    print("  - Expected Weibull severity (constant)")
    print("  - Compound Poisson risk (expected_annual_loss)")
    print("  - Risk percentiles and categories")
    print("  - Spatial geometries for QGIS mapping")

    '''
    Plotting and Summary for diagnostics

    '''

    # Plot counties colored by service area
    plt.figure(figsize=(15, 10))

    scatter = plt.scatter(
        county_data_clean["longitude"],
        county_data_clean["latitude"],
        c=county_data_clean["depot_service_area"],
        cmap="tab10",
        alpha=0.6,
        s=30,
    )

    # Plot depot locations
    plt.scatter(
        depot_df["longitude"],
        depot_df["latitude"],
        c="red",
        marker="*",
        s=200,
        edgecolor="black",
        linewidth=2,
        label="Depot Locations",
    )

    # Annotate depot IDs
    for _, depot in depot_df.iterrows():
        plt.annotate(
            f"D{depot['depot_id']}",
            (depot["longitude"], depot["latitude"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontweight="bold",
        )

    plt.colorbar(scatter, label="Service Area ID")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Optimal Depot Placement ({n_depots} Depots) - Geographic Clustering")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Step 8: Risk vs Distance Analysis
    plt.figure(figsize=(12, 8))
    for depot_id in range(n_depots):
        depot_data = service_stats_df[service_stats_df["depot_id"] == depot_id]
        plt.scatter(
            depot_data["distance_to_depot_km"],
            depot_data["county_risk"],
            label=f"Depot {depot_id}",
            alpha=0.7,
        )

    plt.xlabel("Distance to Depot (km)")
    plt.ylabel("County Risk Level")
    plt.title("Risk vs Distance Analysis by Service Area")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\nModel Summary:")
    print(f"- Created {n_depots} strategic depot locations")
    print(
        f"- Average service radius: {service_stats_df['distance_to_depot_km'].mean():.1f} km"
    )
    print(
        f"- Maximum service distance: {service_stats_df['distance_to_depot_km'].max():.1f} km"
    )
    print(f"- Total counties served: {len(county_data_clean)}")
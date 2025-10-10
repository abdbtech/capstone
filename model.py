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
import db_tools as dbt
from scipy.stats import ks_2samp

# NOTE: Model variables
n_depots = 36  # Set number of depots for clustering analysis

if __name__ == "__main__":

    '''
    Poisson Frequency Component
    '''
    # Load the NOAA data into a DataFrame
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

    # make a dataframe of the poisson risk parameters
    poisson_risk_params = county_lambdas[
        ["county_fips", "lambda_hat", "years_observed", "total_events"]
    ].copy()

    # calculate the probability of at least one event in a year
    poisson_risk_params["prob_at_least_one_event"] = 1 - np.exp(
        -poisson_risk_params["lambda_hat"]
    )
    # Define the columns
    risk_columns = [
        "county_fips",
        "lambda_hat",
        "total_events",
        "years_observed",
        "prob_at_least_one_event"
    ]
    '''
    NRI Shapefile ETL
    '''
    try:
        # Use function to prepare spatial data for QGIS mapping
        lambda_map_clean, db_data = dbt.prepare_spatial_data_for_qgis(
            dataframe=poisson_risk_params,
            geometry_source='nri_shapefile',
            data_columns=risk_columns,
            county_fips_col='county_fips',
            impute_missing=True,
            add_rounded_cols=True
        )
        # Save to database
        dbt.load_data(db_data, "disaster_risk_spatial", if_exists="replace")
        
        # Create spatial table with PostGIS geometry
        spatial_table_sql = """
        DROP TABLE IF EXISTS disaster_risk_counties_spatial;

        CREATE TABLE disaster_risk_counties_spatial AS
        SELECT 
            county_fips,
            lambda_hat_rounded,
            prob_at_least_one_event,
            total_events,
            years_observed,
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

    """
    Table for disaster count by type by county, used for QGIS overlay
    
    Create table with event counts per county
    This table is used for troubleshooting and future functionality
    or for county level historical disaster exploration
    """
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
    # Create the pivot table
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
    # Create the disasters_type_county table for county level event details. 
    # NOTE: Not included in main model but useful for future analysis
    create_enhanced_table_sql = f"""
    DROP TABLE IF EXISTS disasters_type_county;

    CREATE TABLE disasters_type_county AS
    {pivot_sql};

    -- Add indexes
    CREATE INDEX idx_disaster_risk_event_counts_geom ON disasters_type_county USING GIST (geometry);
    ALTER TABLE disasters_type_county ADD CONSTRAINT pk_disaster_risk_event_counts PRIMARY KEY (county_fips);
    """
    dbt.execute_sql(create_enhanced_table_sql)
    '''
    Weibull Severity component
    '''
    # Load tables from database
    df_noaa_episodes = dbt.query('SELECT * FROM "NOAA_STORM_EPISODES"')
    df_census = dbt.query('SELECT * FROM "census_resilience"')

    # Add casualty count to episodes data
    df_noaa_episodes["casualties"] = (
        df_noaa_episodes["total_injuries_direct"]
        + df_noaa_episodes["total_deaths_direct"]
        + df_noaa_episodes["total_injuries_indirect"]
        + df_noaa_episodes["total_deaths_indirect"]
    )
    # Calculate state averages for vulnerability imputation
    df_census["state_fips"] = df_census["County_fips"].str[:2]
    state_vulnerability_avg = df_census.groupby("state_fips")[
        ["PRED12_PE", "PRED3_PE"]
    ].mean()

    # Perform left join to keep all NOAA data. strip state FIPS for averaging later
    noaa_census_full = df_noaa_episodes.merge(
        df_census, left_on="county_fips", right_on="County_fips", how="left" 
        # NOTE: I messed up earlier and called the df_census col 'County_fips' instead of 'county_fips', too late to go back if you value your sanity keep it as it is here :/
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
    # Filter out counties with missing population data instead of imputing to avoid bias
    noaa_census_full = noaa_census_full[noaa_census_full["POPUNI"].notna()]

    # Calculate casualties, casualty rate, vulnerability rate, and intensity
    noaa_census_full["casualties"] = (
        noaa_census_full["total_injuries_direct"]
        + noaa_census_full["total_deaths_direct"]
        + noaa_census_full["total_injuries_indirect"]
        + noaa_census_full["total_deaths_indirect"]
    )
    # calculate casualty rate per 1000 people
    noaa_census_full["casualty_rate"] = (
        noaa_census_full["casualties"] / noaa_census_full["POPUNI"] * 1000
    )
    noaa_census_full["vulnerability_rate"] = (
        noaa_census_full["PRED12_PE_imputed"] + noaa_census_full["PRED3_PE_imputed"]
    )
    # intensity = casualty_rate * (1 + (vulnerability_rate / 100))
    noaa_census_full["intensity"] = noaa_census_full["casualty_rate"] * (
        1 + (noaa_census_full["vulnerability_rate"] / 100)
    )

    # Filter for events with intensity > 0 for distribution fitting
    intensity_data = noaa_census_full[noaa_census_full["intensity"] > 0]["intensity"]

    # Fit distribution to Wiebull using .fit MLE
    weibull_params = weibull_min.fit(intensity_data, floc=0)
    weibull_shape, weibull_loc, weibull_scale = weibull_params

    '''
    Compound Model Integration and Poisson calculation
    '''
    def simulate_compound_poisson_risk(county_fips, n_simulations=10000):
        # Get county's λ (frequency)
        lambda_i = poisson_risk_params[poisson_risk_params["county_fips"] == county_fips][
            "lambda_hat"
        ].iloc[0]

        # Simulate total risk for each year
        total_risks = []
        for _ in range(n_simulations):
            # Generate number of events N ~ Poisson(λ)
            n_events = np.random.poisson(lambda_i)

            # Generate severity for each event Y ~ Weibull
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
    county_risks = {}
    for county in tqdm.tqdm(poisson_risk_params["county_fips"]):
        county_risks[county] = simulate_compound_poisson_risk(county)

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
    # transform to WGS84 to make it work in QGIS
    fix_coordinates_sql = """
    DROP TABLE IF EXISTS disaster_risk_counties_spatial_corrected;
    CREATE TABLE disaster_risk_counties_spatial_corrected AS
    SELECT 
        county_fips,
        lambda_hat_rounded,
        prob_at_least_one_event,
        total_events,
        years_observed,
        ST_Transform(ST_SetSRID(geometry, 3857), 4326) as geometry
    FROM disaster_risk_counties_spatial;

    CREATE INDEX idx_disaster_risk_counties_spatial_corrected_geom 
    ON disaster_risk_counties_spatial_corrected USING GIST (geometry);

    ALTER TABLE disaster_risk_counties_spatial_corrected 
    ADD CONSTRAINT pk_disaster_risk_counties_spatial_corrected PRIMARY KEY (county_fips);
    """
    # extract county centroids and risk data
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
    # Filter out invalid coordinates
    valid_coords = (
        (county_data["latitude"] >= -90)
        & (county_data["latitude"] <= 90)
        & (county_data["longitude"] >= -180)
        & (county_data["longitude"] <= 180)
        & county_data["latitude"].notna()
        & county_data["longitude"].notna()
    )
    # create a cleaned dataframe for clustering
    county_data_clean = county_data[valid_coords].copy()
    
    # Filter out Alaska (02) and Hawaii (15) - focus on contiguous US
    county_data_clean = county_data_clean[
        ~county_data_clean['county_fips'].str.startswith(('02', '15'))
    ].copy()

    # Geographic clustering for depot placement
    geo_features = county_data_clean[["latitude", "longitude"]].values
    geo_scaler = StandardScaler()
    geo_features_scaled = geo_scaler.fit_transform(geo_features)

    # set clusters are run, set var at top with n_depots = 
    # NOTE: future implementation should ask for user prompt for n_depots?
    geo_kmeans = KMeans(n_clusters=n_depots, random_state=36, n_init=10)
    county_data_clean["depot_service_area"] = geo_kmeans.fit_predict(geo_features_scaled)

    # Step 3: Calculate depot locations
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
        # Append depot info
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

    # Save depot locations
    dbt.load_data(depot_df, "strategic_depot_locations", if_exists="replace")

    # Save county assignments to show ovreall clusters
    county_assignments = county_data_clean[["county_fips", "depot_service_area"]].copy()
    county_assignments.rename(columns={"depot_service_area": "depot_id"}, inplace=True)
    dbt.load_data(county_assignments, "county_depot_assignments", if_exists="replace")

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
        r.expected_annual_loss as county_risk,
        g.geometry
    FROM county_depot_assignments c
    JOIN disaster_risk_clusters r ON c.county_fips = r.county_fips
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
        s.lambda_hat_rounded as poisson_frequency,
        s.prob_at_least_one_event,
        s.total_events,
        s.years_observed,
        -- Compound risk metrics
        c.expected_annual_loss as compound_risk,
        c.var_95 as risk_95th_percentile,
        c.var_99 as risk_99th_percentile,
        c.std_dev as risk_std_dev,
        -- Severity component (constant across all counties from fitted distribution)
        {} as expected_severity
    FROM disaster_risk_counties_spatial_corrected s
    JOIN disaster_risk_clusters c ON s.county_fips = c.county_fips
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

    # Impute missing counties with national average
    impute_missing_sql = """
    WITH national_averages AS (
        SELECT 
            AVG(poisson_frequency) as avg_frequency,
            AVG(prob_at_least_one_event) as avg_prob,
            AVG(compound_risk) as avg_risk,
            AVG(risk_95th_percentile) as avg_var95,
            AVG(risk_99th_percentile) as avg_var99,
            AVG(risk_std_dev) as avg_std
        FROM compound_poisson_map
    ),
    missing_counties AS (
        SELECT DISTINCT
            s.county_fips,
            s.geometry
        FROM disaster_risk_counties_spatial_corrected s
        LEFT JOIN compound_poisson_map c ON s.county_fips = c.county_fips
        WHERE c.county_fips IS NULL AND s.geometry IS NOT NULL
    )
    INSERT INTO compound_poisson_map (
        county_fips, geometry, poisson_frequency, prob_at_least_one_event,
        total_events, years_observed, compound_risk, risk_95th_percentile,
        risk_99th_percentile, risk_std_dev, expected_severity
    )
    SELECT 
        m.county_fips,
        m.geometry,
        n.avg_frequency,
        n.avg_prob,
        0,
        0,
        n.avg_risk,
        n.avg_var95,
        n.avg_var99,
        n.avg_std,
        {}
    FROM missing_counties m
    CROSS JOIN national_averages n;
    """.format(expected_severity)
    
    dbt.execute_sql(impute_missing_sql)

    '''
    Clustering Evaluation: Silhouette Score & Gini Coefficient
    '''
    from sklearn.metrics import silhouette_score
    # Calculate silhouette score for geographic clustering
    geo_silhouette = silhouette_score(geo_features_scaled, county_data_clean["depot_service_area"])
    
    # Create risk-weighted clustering for comparison
    risk_geo_features = county_data_clean[["latitude", "longitude", "expected_annual_loss"]].values
    risk_geo_scaler = StandardScaler()
    risk_geo_features_scaled = risk_geo_scaler.fit_transform(risk_geo_features)
    risk_kmeans = KMeans(n_clusters=n_depots, random_state=36, n_init=10)
    county_data_clean["risk_weighted_cluster"] = risk_kmeans.fit_predict(risk_geo_features_scaled)
    
    # Calculate silhouette score for risk-weighted clustering (using geographic features for fair comparison)
    risk_weighted_silhouette = silhouette_score(geo_features_scaled, county_data_clean["risk_weighted_cluster"])

    print(f"Geographic-only clustering silhouette:    {geo_silhouette:.4f}")
    print(f"Risk-weighted clustering silhouette:      {risk_weighted_silhouette:.4f}")
    print(f"Difference:                               {abs(geo_silhouette - risk_weighted_silhouette):.4f}")
    
    # Gini coefficient function
    def gini_coefficient(values):
        """Calculate Gini coefficient for inequality measurement"""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * cumsum[-1]) - (n + 1) / n
    
    # Lorenz curve calculation function
    def lorenz_curve(values):
        """Calculate Lorenz curve coordinates for plotting"""
        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)
        cumsum_pct = cumsum / cumsum[-1]
        pop_pct = np.arange(1, len(values) + 1) / len(values)
        return pop_pct, cumsum_pct
    
    # Calculate risk distribution for geographic clustering
    geo_risk_by_depot = county_data_clean.groupby("depot_service_area")["expected_annual_loss"].sum().values
    geo_gini = gini_coefficient(geo_risk_by_depot)
    geo_pop_pct, geo_cumsum_pct = lorenz_curve(geo_risk_by_depot)
    
    # Calculate risk distribution for risk-weighted clustering
    risk_weighted_risk_by_depot = county_data_clean.groupby("risk_weighted_cluster")["expected_annual_loss"].sum().values
    risk_weighted_gini = gini_coefficient(risk_weighted_risk_by_depot)
    risk_pop_pct, risk_cumsum_pct = lorenz_curve(risk_weighted_risk_by_depot)

    print(f"Geographic-only clustering Gini:          {geo_gini:.4f}")
    print(f"Risk-weighted clustering Gini:            {risk_weighted_gini:.4f}")
    
    # Kolmogorov-Smirnov test for statistical significance
    ks_statistic, ks_pvalue = ks_2samp(geo_risk_by_depot, risk_weighted_risk_by_depot)
    print("Statistical Significance: Kolmogorov-Smirnov Test")
    print(f"KS Test Statistic:                        {ks_statistic:.4f}")
    print(f"P-value:                                  {ks_pvalue:.4f}")

    # Plot Lorenz Curves
    plt.figure(figsize=(10, 8))
    
    # Plot perfect equality line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Equality', alpha=0.5)
    
    # Plot Lorenz curves
    plt.plot(geo_pop_pct, geo_cumsum_pct, linewidth=2.5, 
             label=f'Geographic-Only (Gini={geo_gini:.4f})', color='blue')
    plt.plot(risk_pop_pct, risk_cumsum_pct, linewidth=2.5, 
             label=f'Risk-Weighted (Gini={risk_weighted_gini:.4f})', color='green')
    
    # Fill areas under curves
    plt.fill_between(geo_pop_pct, geo_cumsum_pct, alpha=0.2, color='blue')
    plt.fill_between(risk_pop_pct, risk_cumsum_pct, alpha=0.2, color='green')
    
    plt.xlabel('Cumulative Proportion of Depots', fontsize=12)
    plt.ylabel('Cumulative Proportion of Risk', fontsize=12)
    plt.title('Lorenz Curves: Risk Distribution Equity Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
    print("\n" + "=" * 70)

    '''
    Plotting and Summary for diagnostics
    Use QGIS for detailed spatial analysis using tables created above

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


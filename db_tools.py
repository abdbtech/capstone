
#database utils
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import ftplib
import os
from urllib.parse import urlparse
import ignore.global_vars as gv
import geopandas as gpd
from shapely import wkt
import numpy as np



def get_ftp_filenames(ftp_url, path):
    parsed_url = urlparse(ftp_url)
    ftp = ftplib.FTP(parsed_url.hostname)
    ftp.login()
    ftp.cwd(path)
    files = ftp.nlst()
    ftp.quit()
    return files

def browse_ftp(ftp_url, path="/"):
    """
    Browse FTP directory structure

    ARGS:
        ftp_url (str): FTP server URL
        path (str): Directory path to browse

    RETURNS:
        list: Files and directories
    """
    try:
        parsed_url = urlparse(ftp_url)
        ftp = ftplib.FTP(parsed_url.hostname)
        ftp.login()  # Anonymous login

        ftp.cwd(path)
        items = []
        ftp.retrlines("LIST", items.append)
        ftp.quit()

        print(f"Contents of {path}:")
        for item in items:
            print(item)

        return items
    except Exception as e:
        print(f"Error browsing FTP: {e}")
    
    
# FTP to df utility
def ftp_to_df(ftp_full_url, **kwargs):
    """
    Stream file from FTP server directly to DataFrame using full URL.

    ARGS:
        ftp_full_url (str): Full FTP URL including server, path, and filename
        **kwargs: Additional arguments for pd.read_csv()

    RETURNS:
        pd.DataFrame: Data from FTP file
    """
    try:
        parsed_url = urlparse(ftp_full_url)
        server = parsed_url.hostname
        full_path = parsed_url.path

        # Split the path to get directory and filename
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)

        with ftplib.FTP(server) as ftp:
            ftp.login()  # Anonymous login

            if directory:
                ftp.cwd(directory)

            # Stream directly to memory using BytesIO
            from io import BytesIO

            bio = BytesIO()
            ftp.retrbinary(f"RETR {filename}", bio.write)
            bio.seek(0)  # Reset pointer to beginning

            # Read directly from memory stream into DataFrame
            df = pd.read_csv(bio, **kwargs)
            print(f"Streamed {filename}: {len(df)} rows, {len(df.columns)} columns")
            return df

    except Exception as e:
        print(f"Error streaming from {ftp_full_url}: {e}")
        return pd.DataFrame()
    


# Database connection using psycopg2. Use for direct queries
def connection(test=False):
    """Create a database connection"""
    try:
        conn = psycopg2.connect(
            **gv.localhost
        )

        if test:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            print("Connection established and verified with test")
        else:
            print("Connection established")
        return conn
    except Exception as e:
        print(f"Connection failed: {e}")
        return None
    
# SQLAlchemy engine for pandas and ORM use
def engine():
    """Create a SQLAlchemy database engine"""
    try:
        conn_str = f"postgresql://{gv.localhost['user']}:{gv.localhost['password']}@{gv.localhost['host']}/{gv.localhost['database']}"
        engine = create_engine(conn_str)
        print("Created SQLAlchemy engine for disaster_db")
        return engine
    except Exception as e:
        print(f"Engine creation failed: {e}")
        return None
    
# easy query function to return a dataframe, generates engine internally
def query(sql, params=None):
    """
    Runs a SQL query and returns the results as a DF
    ARGS:
        sql (str): SQL query
        params (tuple, optional): Parameters to include in the query
    RETURNS:
        pd.DataFrame: Query results
    """
    eng = engine()
    if eng:
        try:
            df = pd.read_sql_query(sql, eng, params=params)
            print(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def load_data(data_source, table_name, schema=None, if_exists='replace', **kwargs):
    """
    Load data from a source into a database table.
    ARGS:
        data_source (str or pd.DataFrame): Source of the data (file path or DataFrame)
        table_name (str): Name of the target database table
        schema (str, optional): Schema name (if applicable)
        if_exists (str, optional): Behavior when the table already exists
        **kwargs: Additional arguments passed to pd.to_sql()
    """
    eng = engine()
    if eng:
        try:
            if isinstance(data_source, pd.DataFrame):
                df = data_source
            else:
                df = pd.read_csv(data_source)

            df.to_sql(table_name, eng, schema=schema, if_exists=if_exists, index=False, **kwargs)
            print(f"Data loaded successfully into {table_name}")
        except Exception as e:
            print(f"Error loading data: {e}")

# tool for dropping tables as needed
def drop_table(table_name):
    conn = connection()
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Dropped table {table_name} if it existed.")

# More general tool for executing arbitrary SQL commands
def execute_sql(sql):
    """
    Execute SQL command (for CREATE, DROP, ALTER, etc.)
    ARGS:
        sql (str): SQL command to execute
    """
    conn = connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.commit()
            cursor.close()
            conn.close()
            print("SQL executed successfully")
        except Exception as e:
            print(f"Error executing SQL: {e}")
            if conn:
                conn.close()

def prepare_spatial_data_for_qgis(
    dataframe,
    geometry_source="nri_shapefile",
    data_columns=None,
    county_fips_col="county_fips",
    impute_missing=True,
    add_rounded_cols=True,
):
    """
    Prepare spatial data for QGIS mapping by handling coordinate transformations and data merging.

    Parameters:
    -----------
    dataframe : pd.DataFrame
        DataFrame containing the data to be spatially enabled
    geometry_source : str
        Source of geometry data ('nri_shapefile' or 'existing_spatial_table')
    data_columns : list
        List of columns to include from input dataframe (if None, includes all)
    county_fips_col : str
        Name of the county FIPS column for joining
    impute_missing : bool
        Whether to impute missing values using state averages
    add_rounded_cols : bool
        Whether to add rounded versions of numeric columns for QGIS display

    Returns:
    --------
    tuple: (spatial_dataframe, geometry_wkt_dataframe)
        - spatial_dataframe: GeoDataFrame ready for analysis
        - geometry_wkt_dataframe: DataFrame with WKT geometry for database storage
    """

    if geometry_source == "nri_shapefile":
        # Load NRI shapefile data and convert coordinates
        print("Loading NRI shapefile data...")
        nri_gdf = query("""
            SELECT 
                "STCOFIPS" as county_fips,
                ST_AsText("geometry") as geometry_wkt,
                "TRACTFIPS" as tract_geoid,
                "POPULATION",
                "RISK_SCORE"
            FROM nri_shape_census_tracts
            WHERE "geometry" IS NOT NULL
            AND "STCOFIPS" IS NOT NULL
        """)

        if len(nri_gdf) == 0:
            raise ValueError("No NRI shapefile data available")

        print(f"Loaded {len(nri_gdf)} census tracts")

        # Convert WKT to geometries for processing
        nri_gdf["geometry"] = nri_gdf["geometry_wkt"].apply(wkt.loads)
        nri_gdf = gpd.GeoDataFrame(nri_gdf, geometry="geometry")

        # Aggregate to county level
        print("Aggregating to county boundaries...")
        county_boundaries = nri_gdf.dissolve(by="county_fips").reset_index()
        county_boundaries = county_boundaries[["county_fips", "geometry"]]

    elif geometry_source == "existing_spatial_table":
        # This would handle the coordinate correction case
        print("Using existing spatial table with coordinate correction...")
        # Note: This path would be implemented when we have an existing spatial table
        # that needs coordinate system correction
        raise NotImplementedError("Existing spatial table source not yet implemented")

    else:
        raise ValueError(f"Unknown geometry_source: {geometry_source}")

    # Prepare data columns
    if data_columns is None:
        data_columns = list(dataframe.columns)

    # Ensure county_fips_col is included
    if county_fips_col not in data_columns:
        data_columns.append(county_fips_col)

    # Merge with input data
    print("Merging with input data...")
    spatial_data = county_boundaries.merge(
        dataframe[data_columns],
        left_on="county_fips",
        right_on=county_fips_col,
        how="left",
    )

    # Handle missing data imputation
    if impute_missing:
        print("Imputing missing values using state averages...")
        spatial_data["state_fips"] = spatial_data["county_fips"].str[:2]

        # Identify numeric columns for imputation (excluding ID columns)
        numeric_cols = spatial_data.select_dtypes(include=[np.number]).columns
        exclude_cols = ["county_fips", "state_fips"]
        impute_cols = [col for col in numeric_cols if col not in exclude_cols]

        for col in impute_cols:
            if spatial_data[col].isnull().any():
                # Calculate state averages
                state_avg = spatial_data.groupby("state_fips")[col].mean()
                # Fill missing values with state averages
                spatial_data[col] = spatial_data[col].fillna(
                    spatial_data["state_fips"].map(state_avg)
                )
                # Fill any remaining nulls with overall mean
                overall_mean = spatial_data[col].mean()
                spatial_data[col] = spatial_data[col].fillna(overall_mean)

    # Add rounded columns for better QGIS display
    if add_rounded_cols:
        numeric_cols = spatial_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["county_fips", "state_fips"] and not col.endswith(
                "_rounded"
            ):
                # Determine appropriate rounding based on magnitude
                col_values = spatial_data[col].dropna()
                if len(col_values) > 0:
                    max_val = col_values.abs().max()
                    if max_val >= 1000:
                        decimals = 0
                    elif max_val >= 10:
                        decimals = 1
                    elif max_val >= 1:
                        decimals = 2
                    else:
                        decimals = 3

                    spatial_data[f"{col}_rounded"] = spatial_data[col].round(decimals)

    print(f"Prepared {len(spatial_data)} counties with complete spatial data")

    # Create WKT version for database storage
    geometry_wkt_data = spatial_data.copy()
    geometry_wkt_data["geometry_wkt"] = geometry_wkt_data["geometry"].apply(
        lambda x: x.wkt
    )

    # Drop the geometry column for database storage
    db_data = geometry_wkt_data.drop("geometry", axis=1)

    return spatial_data, db_data
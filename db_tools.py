
#database utils
import psycopg2
from sqlalchemy import create_engine
import pandas as pd
import ftplib
import os
from urllib.parse import urlparse
import ignore.global_vars as gv



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
    


# Database connection. Right now this is on portia, update to fabian when we can
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

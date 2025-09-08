#!/usr/bin/env python3
"""
Load NRI (National Risk Index) data into PostgreSQL database
for the Disaster Response Capstone Project
"""

from data.database_setup import load_nri_shapefile, setup_project_schema
import os

def main():
    """Load NRI data into the disaster response database"""
    
    # Common download locations - update these paths as needed
    possible_paths = [
        r"C:\Users\alben\Downloads\NRI_Shapefile_Counties.zip",
        r"C:\Downloads\NRI_Shapefile_Counties.zip", 
        r".\data\NRI_Shapefile_Counties.zip",
        # Add your actual download path here
    ]
    
    print("=== NRI Data Loader for Disaster Response Project ===")
    print("\nLooking for NRI shapefile...")
    
    # Find the NRI file
    nri_file = None
    for path in possible_paths:
        if os.path.exists(path):
            nri_file = path
            print(f"✓ Found NRI file: {path}")
            break
    
    if not nri_file:
        print("✗ NRI shapefile not found in common locations.")
        print("\nPlease download from: https://www.fema.gov/flood-maps/products-tools/national-risk-index")
        print("Or enter the path manually:")
        nri_file = input("Enter path to NRI shapefile (ZIP or .shp): ").strip()
        
        if not os.path.exists(nri_file):
            print(f"✗ File not found: {nri_file}")
            return False
    
    # Load the data
    success = load_nri_shapefile(nri_file, 'raw_data.nri_counties')
    
    if success:
        print("\n🎉 NRI data loaded successfully!")
        print("\nYour disaster response database now contains:")
        print("  - Table: raw_data.nri_counties (National Risk Index data)")
        print("  - Geographic data for risk analysis and modeling")
        print("  - Ready for your mutual aid supply cache optimization!")
        
        # Next steps
        print("\n=== Next Steps ===")
        print("1. Load FEMA disaster declarations data")
        print("2. Add Census demographic data") 
        print("3. Begin geographic risk profiling")
        print("4. Develop supply cache optimization model")
        
    else:
        print("✗ Failed to load NRI data")
        
    return success

if __name__ == "__main__":
    main()

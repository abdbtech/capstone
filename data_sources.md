Data:
https://public.emdat.be/
EM-DAT contains data on the occurrence and impacts of over 26,000 mass disasters worldwide from 1900 to the present day.
(data downloaded for US on 8/30)
https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.Analyses/.SPI/?sem=iridl%3AClimate-Indices
Standardized Precipitation Index over previous 45 years.
(need to re-check, might not be usable)
https://wisqars.cdc.gov/
Injury and death data 
(dashboards but haven’t found any clean datasets yet)

https://www.earthdata.nasa.gov/topics/human-dimensions/natural-hazards/data-access-tools#toc-natural-hazards-datasets
Geocoded Disasters (GDIS) Dataset
Global Fire Atlas with Characteristics of Individual Fires, 2003-2016
NAFD-ATT Forest Canopy Cover Loss from Landsat, CONUS, 1986-2010

https://www.ncei.noaa.gov/stormevents/ftp.jsp
Storm event dataset

https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0209268
Economic impact data, might not use



### promising leads ###

fema
https://www.fema.gov/about/reports-and-data/openfema
API access
https://www.fema.gov/about/openfema/api
(https://www.fema.gov/api/open)

fema risk index
https://hazards.fema.gov/nri/data-resources

    shapefile for all counties
    downloaded: "C:\Users\alben\Downloads\NRI_Shapefile_Counties.zip"

noaa
storms
https://www.ncei.noaa.gov/stormevents/

    3 ftp access points

hurricanes
https://oceanservice.noaa.gov/news/historical-hurricanes/


census 
api access
https://www.census.gov/data/developers/data-sets.html

Variable Description	Rationale for Inclusion	Corresponding ACS Table
Median Household Income	Represents economic vulnerability. Lower income areas may have fewer resources for preparedness.	

B19013  

Age Distribution (e.g., % 65+)	Senior populations may face increased vulnerability and mobility challenges during a disaster.	

B01002  

Disability Status	Individuals with disabilities may have specific needs during an emergency and require additional assistance.	

B18101  

Housing Occupancy & Type	Indicators for housing density, property value, and construction type.	

B25002, B25034  

Vehicle Availability	Car ownership rates can serve as a proxy for evacuation capability.	

B08006  

Poverty Status	Communities with high poverty rates may have less capacity to recover from a disaster.	

S1701   







### Partially implemented ###

datasets explored further

## fema risk index ##
1. FEMA NRI
     https://hazards.fema.gov/nri/data-resources

   ~~ shapefile for all counties
    downloaded: "C:\Users\alben\Downloads\NRI_Shapefile_Counties.zip"
    NRIDataDictionary.csv~~

    Using census tract dataset instead for higher resolution:
    https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload//NRI_Shapefile_CensusTracts/NRI_Shapefile_CensusTracts.zip

    
## Billion dollar storms ##

https://www.ncei.noaa.gov/access/billions/


## Census comunity resiliance estimates ##

https://www.census.gov/programs-surveys/community-resilience-estimates/data/datasets.html



### In SQL and tested ###

1. FEMA NRI
    NRI_shapefile_censustracts.zip as TABLE 'nri_shape_census_tracts'

2. NOAA Storm events
    27 files covering 1999-2025 as NOAA_STORM_EVENTS

3. 
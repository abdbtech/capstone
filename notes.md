The purpose of this document is to help you clearly explain your capstone topic, project scope, and timeline. Identify each of the following areas so you will have a complete and realistic overview of your project. Your course instructor cannot approve your project topic without this information.

Student Name:  Alex Barton	

Student ID: 011318156

Capstone Project Name: 
Modeling the geographic risk profile of disasters using statistical methods to assist in the allocation of resources by decentralized mutual aid disaster response groups.

Project Topic: 
This project will use publicly available data to analyze the risk of disaster frequency and severity by geographical area in order to provide mutual aid groups with logical locations to deploy supply caches and depots. 

Research Question: 
How can the statistical analysis and modeling of publicly available datasets related to disasters assist in the efficient staging of supplies prior to disasters occurring.


Hypothesis: 
An analysis of historical disaster data in the continental US can be used to determine optimal locations to position supplies and equipment for use in a future disaster.

Disasters are unpredictable by nature and the validation of this model will require simulated testing using both test data derived from historical events and silhouette validation to ensure the clusters are properly classified. 


Context: 
Mutual aid groups operate in a non hierarchical manner with limited financial resources. While adhoc responses to disasters are effective and necessary, a method of determine the optimal placement of limited resources is a topic often discussed and desired. Providing an open source model based on public data will allow for local analysis when determining the optimal location for shared resources. 





Data Gathering: 
All datasets above are publicly available and accessible via FTP and direct download. Beginning in 2025, datasets related to natural disasters and climate have been removed from government databases at random. All datasets identified above will be archived locally to minimize the risk of data deletion. Additional datasets may be identified and integrated. 


Data Analytics Tools and Techniques: 
Tools:
Python, R, Git

Techniques:
Weibull analysis will be used to model the probability distribution of events for granular geographical areas (County level would be acceptable). This, along with other parameters, will be used to create a risk profile for each geographic area for use in the machine learning model.
Clustering models, likely K-means, will be trained on the statistical parameters, geographical data and potentially other data to produce clusters of candidate locations. 
Silhouette analysis will be performed to ensure the clusters are appropriately generated.
A two-sample t-test will be used to determine if the model placement performs better than random placement using simulated data based on real datasets. 

Justification of Tools/Techniques: 
Weibull analysis was chosen because it can accurately model time to events which is essential for this project.
Clustering models were chosen because the desired output of the model is a series of spacial locations, clustering will provide the most accurate modeling. As an unsupervised learning technique, clustering groups similar data with a centroid tha will aid with optimal placement.
Silhouette analysis was chosen because clustering models will be used, this method demonstrates the accuracy and quality of the clustering model.
Two-sample t-test was chosen because a performance comparison using random data is required. Using the resulting p value and significance level the hull hypothesis can be accurately evaluated. 

Application Type, if applicable (select one):
☐ Mobile
☐ Web
X Stand-alone

Programming/Development Language(s), if applicable: 
R, Python


Operating System(s)/Platform(s), if applicable: 
WIN11, possibly docker and linux


Database Management System, if applicable: 
None foreseen but postgresql if required.

Project Outcomes: 
The project will be considered a success if a validated graphical map with projected locations of optimal supply locations is generated. 

Projected Project End Date: 
30SEP-30OCT

Sources: 
NA



Postgresql running with postgis

server
MADR
password
cmos555

databases:
postgis

to make requirement doc automatically:
pip freeze > requirements.txt



FIPS

FIPS format (11 digits): SSCCCTTTTTT

SS = 2-digit state code
CCC = 3-digit county code
TTTTTT = 6-digit (census) tract code
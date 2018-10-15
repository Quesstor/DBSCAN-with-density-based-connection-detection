# DBSCAN-with-density-based-connection-detection
Density based clustering methods are able to identify areas with dense clusters
of any shape and size. DBSCAN is one of the basic methods in this group, but
clustering with DBSCAN has a major drawback if the clusters in question are
connected by small chains. Imagine two separated clouds of points connected
by a small chain of points. Intuitively these two clouds are different clusters,
but DBSCAN may detect these as a single huge one, because the algorithm
expands the cluster along these chains. This is sometimes called the single
link effect. 

This algorithm detects these chains in multidimensional data spaces.

For more explanations and examples see Thesis.pdf

# Dataset (accidents_2012_to_2014.csv)
The dataset was downloaded on February the 27th 2018 from https://www.kaggle.com/daveianhickey/2000-16-traffic-flow-england-scotland-wales/data. 
The license for this dataset is the Open Government Licence used by all data on data.gov.uk (http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/). 
The raw datasets are available from the UK Department of Transport website https://www.dft.gov.uk/traffic-counts/download.php

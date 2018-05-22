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

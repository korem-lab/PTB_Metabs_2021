import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import kneighbors_graph, DistanceMetric
from get_dfs import get_metabs
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

def figure_S3():

	def generate_random_reference_uniform(df):
	    rd_features=[]
	    for i in range(len(df.iloc[0,:])):
	        a = np.min(df.iloc[:,i])
	        b = np.max(df.iloc[:,i])
	        rd_features.append(np.random.uniform(a,b,len(df)))
	    rd_df = pd.DataFrame(data=np.array(rd_features).T, columns=df.columns, index=df.index)
	    return rd_df
	
	# specify parameters
	kmax = 15
	internal_inertia = True

	df = get_metabs(rzscore=True, min_impute=True)
	dist = pairwise_distances(df, metric="canberra")

	wcss = []
	gaps = []
	ch_index = []
	sil = []

	for k in range(1, kmax+1):
	    cluster = KMedoids(n_clusters=k, metric="precomputed", random_state=42).fit(dist)
	    curr_wcss = cluster.inertia_
	    wcss.append(curr_wcss)
	    
	    # generate random distributions and cluster in order to calculate gap stat
	    rd_wcss = []
	    for ref_num in range(10):
	        rd_curr_wcss = 0
	        rd_df = generate_random_reference_uniform(df)
	        rd_dist = pairwise_distances(rd_df, metric='canberra')
	        rd_cluster = KMedoids(n_clusters=k, metric="precomputed", random_state=42).fit(rd_dist)
	        rd_curr_wcss = rd_cluster.inertia_ # use the distance metric passed in cluster
	        rd_wcss.append(rd_curr_wcss)

	    ### calculate the gap statistic for current k
	    gap = np.log(np.mean(rd_wcss)) - np.log(curr_wcss)
	    gaps.append(gap)


	plt.figure(figsize=(8,4))
	plt.plot(list(range(1,kmax+1)), wcss, '-o')
	plt.xticks(list(range(1,kmax+1)))
	plt.savefig("figurePanels/S3A.png", dpi=800, bbox_inches='tight')
	plt.savefig("figurePanels/S3A.pdf", dpi=800, bbox_inches='tight')
	plt.figure(figsize=(8,4))
	plt.plot(list(range(1,kmax+1)), gaps, '-o')
	plt.xticks(list(range(1,kmax+1)))
	plt.savefig("figurePanels/S3B.png", dpi=800, bbox_inches='tight')
	plt.savefig("figurePanels/S3B.pdf", dpi=800, bbox_inches='tight')
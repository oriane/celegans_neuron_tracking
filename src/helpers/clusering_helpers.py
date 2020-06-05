import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import pickle

import gmmreg
import gmmreg._core as core
####################################################################################################################

def center_match(scene,model):
    """
    Performs gmmreg on scene using model as reference.
    INPUT: Scene and model are numpy array of each centers of segement each for one volume of the video
    OUTPUT:  Set Registration of scene unto model (see paper below for detail) as a numpy array
        
    The code is adapted from :

    @article{Jian&Vemuri_pami11,
    author  = {Bing Jian and Baba C. Vemuri},
    title   = {Robust Point Set Registration Using {Gaussian} Mixture Models},
    journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
    year = {2011},
    volume = {33},
    number = {8},
    pages = {1633-1645},
    url = {https://github.com/bing-jian/gmmreg/},
    }

    The settings are the ones used in the exemple of the paper.
    """
    # Add setting as specified in the test 
    ctrl_pts = model
    level = 4
    scales = [.3,.2,.1,.05]
    lambdas = [.1,.02,.01,.02,0,0,0]
    iters = [100,200,300,400]
    normalize_flag = 1
    
    # perform matching
    if normalize_flag==1:
        [model, c_m, s_m] = core.normalize(model)
        [scene, c_s, s_s] = core.normalize(scene)
        [ctrl_pts, c_c, s_c] = core.normalize(ctrl_pts)
    
    after_tps = core.run_multi_level(model,scene,ctrl_pts,level,scales,lambdas,iters)
    
    if normalize_flag==1:
        model = core.denormalize(model,c_m,s_m)
        scene = core.denormalize(scene,c_s,s_s)
        after_tps = core.denormalize(after_tps,c_s,s_s)
    return after_tps

####################################################################################################################

def get_features(df,nfeats = 10):
    """
    Perform set registration between every time frames and the reference time frames. 
    The output is the projection of each segements of a frame unto a reference volume. 
    This output is then used as the features for clustering. 
    Each references volume produce 3 features (one for each dimensions).
    INPUT : 
        - df : the processed resuts of segmentation
        - nfeats : the number of reference volume to use
    OUTPUT :
        - A df containing the reference volume (useful for error correction)
        - A new df containing the features of each segment added
   """
    # Pick at regular interval the number of features we want
    new_df = df.copy()
    index_list = df["Time"].unique()
    refs_index = np.linspace(0,len(index_list),nfeats,endpoint = False)
    refs = index_list[list(map(int, refs_index))]
    refs_df = df.loc[df.Time.isin(refs)]
    print("the number of feature is :", len(refs))
    
    matches = [] # Register in format needed for clustering
    for i,volume in df.groupby("Time"):
        feats = []
        
        # Get centroids of volume
        v = np.stack(volume.centroids.values)
        
        for j,ref in refs_df.groupby("Time"):
            
            # Get centroids of reference
            r = np.stack(ref.centroids.values)
            
            # Perform matching
            feats.append(center_match(r,v))
        
        # Restruture data so we get all features for each segment
        aa = np.array(feats).transpose((1,0,2))
        [matches.append(a) for a in aa]
    
    # Data restructuration continued
    asarray = np.stack(matches)
    x,y,z = asarray.shape
    result = asarray.reshape(x,y*z)
    # add the features to the new data frame
    new_df["Features"] = [result[i,:] for i in range(result.shape[0])]
    return refs_df,new_df

####################################################################################################################

def get_clusters(matches,ncluster):
    """ 
    Perform KMeans on the features of the each segment of the video.
    INPUT : 
        - matches : The features as produced by *get_features* 
        - ncluster : the number of cluster
    OUTPUT: The trained kmeans classifier 
    """
    
    clust = KMeans(n_clusters=ncluster)
    
    clust.fit(matches)
    
    return clust

####################################################################################################################

def error_correction(df,classifier):
    """
    Remove cluster duplicates in a same volume. 
    When collision occurs the segment the closest to the cluster is kept, the rest are reassigned. 
    Reassigment is done based on the rank of closness to the cluster. 
    This means that the segment that at any time frame the nth segement closest to the ith cluster is reassigned to the same new cluster.  
    INPUT : 
        - df with the assigned cluster
        - Classifier : trained KMeans classifier
    OUTPUT : df without cluster collision 
    """
    affinity = classifier.fit_transform(np.stack(df.centroids))
    df["dist"] = affinity[np.arange(len(affinity)),df["Clusters"]]
    crossed = df.set_index(['Time', 'Clusters']).join(df.groupby(["Time","Clusters"]).agg({"dist":min}), rsuffix='_min').reset_index()
    c = crossed.sort_values(['Time', 'Clusters', 'dist'])
    c_ = c.groupby(['Time', 'Clusters']).agg({'id': lambda v: tuple(range(len(v)))})
    c_ = c_.explode('id')
    c_ = c_.reset_index()
    c['Clusters'] = c_['Clusters'] * 100 + c_['id']
    rename = {c: i for i, c in enumerate(sorted(set(c['Clusters'])))}
    c['Clusters'] = c['Clusters'].replace(rename)
    return c

def neurons_tracking(df,filename = ""):
    """ 
    Perform clustering of segmented neurons
    INPUT :
        - df is the DataFrame containing the result of segmentation of the c.elegans recording
        - filename : if a filename is passed, the corrected at ../produced/{filename}_clusters.p
        
    OUTPUT :
        - refs_df : a DataFrame containing all the reference volumes (usefull in error corrections)
        - corrected :  The DataFrame containing the Classification results
    """
    
    # Set parameters
    nfeats = 10 # Number of reference volume to use
    factor = 1.2 # Fraction of cluster to use compared to the maxiumum amount of segment in a single volume 
    
    n_cluster = int(df.groupby("Time").count().max()[0])
    
    # Peforms set registration 
    refs_df,new_df =  get_features(df,nfeats)
    # Create an array with features to feed to the clustering methods
    matches = np.stack(new_df.Features.values)
    
    # Adding the segmented centroids to the features and putting a g^higger weight on it
    data = np.hstack((matches,np.stack(df.centroids)))
    classifier = get_clusters(data,n_cluster)
    new_df["Clusters"] = classifier.labels_

    # Reassign the clusters that are duplicate in a time frame
    corrected = error_correction(new_df,classifier)
    
    if filename:
        with open(f'../produced/{filename}_clusters.p', 'wb') as f:
            pickle.dump(corrected ,f)
        print(f'Saved at ../produced/{filename}_clusters.p' )
    return refs_df,corrected,new_df,classifier

def update_cluster(df,old_cluster,new_cluster):
    """ Reassigned all the segment previously assigned to the olf_cluster to new_cluster
        INPUT : 
            - df : the df resulting from neuron_tracking
            - old_cluster : the id of the cluster to be changed
            - new_cluster : id of the target cluster
        OUTPUT :
            - None, df is updated"""
    change_idx = df.loc[df.Clusters == old_cluster].index
    df.loc[change_idx,"Clusters"] = new_cluster


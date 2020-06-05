import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import pickle

def get_centroids(x,y,z):
    """
    Get centroid of a 3d cloud of points
    INPUT: 3 lists one for each of the x,y,z coordinates
    OUTPUT: 3d coordinated of the centroids of the clouds
    
    """
    
    return np.array((sum(x) / len(x), sum(y) / len(y), sum(z) / len(z)))

def get_contours(df):
    """
    Finds the contours of the segment contained in df. The contours are determined as the vertice of the convex hull of each segment.
    INPUT : A df containing the segments and a 3 lists of coordinates, one for of each x,y,z. 
    OUTPUT : The contours as a array. 
    
    """
    # check if a dim is constant as hull would fail
    is_cst = np.stack(df[["x","y","z"]]).min(axis=1) == np.stack(df[["x","y","z"]]).max(axis=1)
    # get the index of the cst dim if it exists
    ind_cst = np.where(is_cst)[0]
    if ind_cst.size != 0:
        ind_cst = ind_cst[0]
        to_get = ["x","y","z"]
        cons_val = df[to_get[ind_cst]][0]
        to_get.remove(to_get[ind_cst])
        points = np.stack(df[to_get]).T
        val = points[ConvexHull(points).vertices]
        return np.concatenate([val,np.array([cons_val]*len(val))[:, np.newaxis]], axis=1)
        
    else:
        points = np.stack(df[["x","y","z"]]).T
        return  points[ConvexHull(points).vertices]
        

def load_files(filename, already_clustered = False, size_threshold = 10,
               add_assigment = False, filter_time = True,get_contour=True):
    """
    Loading the file for neurons tracking. This function expect the files resulting from the segmentation under ../data and the result of a previous clustering under ../produced.
    INPUT:
        - filename: name of the file to load
        - already_clustered: if True, load a pickle file that already has been clustered
        - size_threshold: for size filtering, what size percentile to remove (for no filtering put 0)
        - add_assigment: Join with the assigment df to compare with previous clustering
        - filter_time: Remove every two frame to compensate for camera movement
        - get_contour :  Calculate the contours of each segment (for plotting).
    OUTPUT:
        - Either clustered or segemented version of the video
    
    """
    if already_clustered:
        with open(f'../produced/{filename}_clusters.p', 'rb') as f:
            clustered_df = pickle.load(f)
        return clustered_df
    else:
        # Get the results of the segmentation
        segmented_df = pd.read_csv(f"../data/{filename}_segmented.csv")
        
        # Remove one every two frame to compensate camera movement
        if filter_time:
            segmented_df  = segmented_df.loc[segmented_df.Time%2 != 0]
        
        # Group all coordinate together in a list so one row in the df is one segement
        segmented_df = segmented_df.groupby(["Time","Segment"]).agg({'x':list, 'y':list, 'z':list})
            
        # Find the centers of each segment
        segmented_df['centroids'] = segmented_df[['x','y','z']].apply(lambda d: get_centroids(d.x,d.y,d.z),axis=1)
        
        # Get the assignement of the old clustering methods. 
        if add_assigment:
            assignment = pd.read_csv(f"../data/{filename}_assignment.csv")
            if filter_time:
                assignment  = assignment.loc[assignment.Time%2 != 0]
            segmented_df = assignment.join(segmented_df,on=["Time","Segment"])
        segmented_df = segmented_df.reset_index()
        
        # keep an id for tracability trough tranformation
        segmented_df["id"] = segmented_df.index
        
        # Keep the contours of the segments
        if get_contour:
            
            segmented_df["contours"] = segmented_df[["x","y","z"]].apply( get_contours,axis=1)
        return segmented_df

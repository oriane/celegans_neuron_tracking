import numpy as np
from collections import Counter

from mpl_toolkits import mplot3d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import ipywidgets as ipyw

import imageio


def plot_clustering(to_plot,to_color):
    """ 
    Plot the df given as to_plot, color the cluster as specified by to_color
    INPUT: a DataFrame with centroids to plot and a column to_color as a clustering, 
        to_color is a string representing the columns of to_plot to use for cluster coloring
    OUTPUT: None 
    
    """
    plt.figure(figsize=(6,6))
    ax = plt.axes( projection='3d')
    pts = np.stack(to_plot["centroids"].values)
    # Plot a color per clusters
    colors = to_plot[to_color].values
    # plot
    scatter = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors,cmap = 'tab20',alpha=0.5)
    
########################################################################################################

def plot_interactive(to_plot,to_color):
    """ 
    Interactive plot that allows to view the cluster individually.
    INPUT: a DataFrame with centroids to plot and a column to_color as a clustering, 
        to_color is a string representing the columns of to_plot to use for cluster coloring
    OUTPUT: None 
    
    """
    plt.figure(figsize=(5,5))
    minx,maxx = get_lim(to_plot,'x')
    miny,maxy = get_lim(to_plot,'y')
    minz,maxz = get_lim(to_plot,'z')
    
    
    # Make a list of the clusters we want
    clusters = to_plot[to_color].unique()
    
    def update(i):
        ax = plt.axes( projection='3d')
        ax.set_xlim([minx - 10 ,maxx + 10])
        ax.set_ylim([miny - 10  ,maxy + 10])
        ax.set_zlim([minz - 10 ,maxz + 10])
        df = to_plot.loc[to_plot[to_color] == clusters[i]]
        pts = np.stack(df["centroids"].values)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='k')
    
    ipyw.interact(update,i=(0,len(clusters)-1));

########################################################################################################

def get_lim(df,axis):
    """
    Get the minimum and maxiumum values of a given axis of the df
    INPUT: Df to plot and axis whose limit is sought
    OUTPUT: min of axis, maxiumum of axis
    
    """
    all = df[axis].explode()
    return all.min(),all.max()

########################################################################################################

def plot_tru_time(to_plot):
    """ 
    Interactive plot that allows to view the segement trought time. Two clusters can be seen in parallel, to see only one set c2 to 999.
        INPUT: a DataFrame with centroids to plot and the New_Clusters column
        OUTPUT: None 
    """
    plt.figure(figsize=(5,5))
    minx,maxx = get_lim(to_plot,'x')
    miny,maxy = get_lim(to_plot,'y')
    minz,maxz = get_lim(to_plot,'z')
    
    def update(t,c,c2):
        ax = plt.axes( projection='3d')
        ax.set_xlim([minx - 10 ,maxx + 10])
        ax.set_ylim([miny - 10  ,maxy + 10])
        ax.set_zlim([minz - 10 ,maxz + 10])
        
        tmp = to_plot.loc[to_plot["Time"] == 2*t+1]
        df = tmp.loc[tmp.New_Clusters == c]
        for k,v in df.iterrows():
            x = np.stack(v.x)
            y = np.stack(v.y)
            z = np.stack(v.z)
            ax.scatter(x,y,z)
        if c2 != 999 :
            df_2 = tmp.loc[tmp.New_Clusters == c2]
            for k,v in df_2.iterrows():
                x = np.stack(v.x)
                y = np.stack(v.y)
                z = np.stack(v.z)
                ax.scatter(x,y,z)
    
    ipyw.interact(update,t=(0,len(to_plot.groupby("Time"))-1),c = to_plot.New_Clusters.unique(), c2 = [999.0] + list(to_plot.New_Clusters.unique())  );
    
########################################################################################################
    
def get_contour(df):
    """
    Get the contour of the segement of the pipeline segmentation
    INPUT: The df containing the segmenattion at one zlevel
    OUTPUT: A list of 2d coordinate containing the rought contours of the segment 
    """
    
    onx = df.groupby(['Segment','x'])
    ony = df.groupby(['Segment','y'])
    tmp1 = onx.max().groupby('Segment').count()
    tmp2 = ony.max().groupby('Segment').count()
    first_val = np.hstack([[k]*v.Time for k,v in tmp1.iterrows()]*2)
    scd_val = np.hstack([[k]*v.Time for k,v in tmp2.iterrows()]*2)
    vals =  np.hstack((first_val,scd_val))
    maxx = np.stack((onx.max().y.index.get_level_values(1).values,onx.max().y.values),1)
    minx = np.stack((onx.min().y.index.get_level_values(1).values,onx.min().y.values),1)
    maxy = np.stack((ony.max().x.values,ony.max().x.index.get_level_values(1).values),1)
    miny = np.stack((ony.min().x.values,ony.min().x.index.get_level_values(1).values ),1)
    return np.vstack((maxx,minx,maxy,miny)),vals

########################################################################################################

def nd2_read(filename,df,path):
    """ 
    Interactive plot allowing to superpose segment assigment to nd2 video.
    INPUT: 
        - filename : name of the file. 
        - df : the df containing the result of neuron_tracking 
        - path: path to the nd2 tiff images.
    OUTPUT: None 
    
    """
    def update(t,z,c):
        plt.figure(figsize=(9,9))
        plt.grid(False)
        # Plot only the odd numbers
        t = float(2*t+1)
        # the files start at 1 whereas the df starts at 0 so we must add 1 for matching
        time = "%.3d" % float(t+1)
        zlevel = "%.2d" % z
        im = imageio.imread(f'{path}/{filename}_exported/{filename}T{time}Z{zlevel}C{c}.tif')
        background = Counter(im.flatten()).most_common(1)[0][0]
        plt.imshow(im/background)
        df_t = df.loc[df.Time == t].copy()
        df_t['contours'] =df_t["contours"].map(lambda contours: np.array([c for c in contours if c[-1] == z]))
        df_t = df_t[df_t['contours'].map(len) != 0]
        if len(df_t) != 0 :
            means = df_t["contours"].apply(lambda a:np.array([np.mean(a[:,0]),np.mean(a[:,1])]))
            xs,ys = np.stack(means)[:,0],np.stack(means)[:,1]
            colors = df_t.Clusters.values
            contours = np.vstack(df_t.contours)
            plt.scatter(contours[:,1],contours[:,0],c= 'w',s=15,marker ='x',alpha=0.1)
            for x,y,c in zip(xs,ys,colors):
                plt.scatter(y,x,c= 'c',s=150,marker = f"${int(c)}$")
    ipyw.interact(update,t=(0,int(df.Time.max()/2)),z=(1,34),c=[2,1]  );

########################################################################################################
    
def plot_through_time(df):
    """
    Shows evolution of center of segments through time.
    INPUT : df resulting from the segmentation
    OUTPUT : None
    
    """
    for_viz = df.copy().reset_index()
    fig = plt.figure(figsize=(7,6))
    ax = plt.axes(projection='3d')
    pts = np.stack(for_viz.centroids.values)
    times = for_viz.Time.values
    h = ax.scatter(pts[...,0], pts[...,1], pts[...,2],c =times,cmap = 'RdPu')
    fig.colorbar(h,ax=ax, label = "time")
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.set_title("Centroids through Time")
    
########################################################################################################

def plot_features(featured):
    """
    Shows the center of the segment of the first frame and the features it's mapped to.
    INPUT : df resulting from clustering_helpers.get_feature()
    OUTPUT : None
    
    """
    plt.figure(figsize=(8,8))
    ax = plt.axes( projection='3d')
    single_pt = featured.iloc[0].centroids
    features = np.stack(featured.loc[featured.Time == 1].Features).reshape(320,3)
    segments = featured.loc[featured.Time == 1].Segment
    ax.scatter(features[:,0], features[:,1], features[:,2],c = np.repeat(segments,10), cmap = "tab20",alpha=0.2,marker="x")
    other_points = np.stack(featured.loc[featured.Time == 1].centroids)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    ax.scatter(other_points[:,0], other_points[:,1], other_points[:,2],c = segments, cmap = "tab20",alpha=1)
    
########################################################################################################

def view_clusters(df):
    """
    3D view of the clusters with chosable time.
    INPUT : df resulting from clustering_helpers.neurons_clustering()
    OUTPUT : None
    
    """
    plt.figure(figsize=(4,4))
    def update(t):
        ax = plt.axes( projection='3d')
        t = t*2+1
        df_i = df.loc[(df.Time == t)]
        n_clusters = len(df_i.Clusters.unique())
        all_colors = np.array(list(mcolors.CSS4_COLORS.keys()))
        colors = all_colors[np.linspace(0,len(all_colors)-1,n_clusters, dtype=int)]
        for i,(k,v) in enumerate(df_i.iterrows()):
            pts = v.contours
            ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors[i])

    ipyw.interact(update,t=(0,int(df.Time.max()/2)));
    
########################################################################################################
    
def view_two_clusters(df):
    """
    3D view of the clusters with chosable time.
    INPUT : df resulting from clustering_helpers.neurons_clustering()
    OUTPUT : None
    
    """
    fig = plt.figure(figsize=(4,4))
    minx,maxx = get_lim(df,'x')
    miny,maxy = get_lim(df,'y')
    minz,maxz = get_lim(df,'z')
    def update(t,c,c2):
        ax = plt.axes( projection='3d')
        ax.set_xlim([minx - 10 ,maxx + 10])
        ax.set_ylim([miny - 10  ,maxy + 10])
        ax.set_zlim([minz - 10 ,maxz + 10])
        t = t*2+1
        df_i = df.loc[(df.Time == t) & (df.Clusters == c)]
        df_2 = df.loc[(df.Time == t) & (df.Clusters == c2)]
        n_clusters = len(df_i.Clusters.unique())
        all_colors = np.array(list(mcolors.CSS4_COLORS.keys()))
        colors = all_colors[np.linspace(0,len(all_colors)-1,n_clusters, dtype=int)]
        for color,df_ in zip(['g','c'],[df_i,df_2]):
            for i,(k,v) in enumerate(df_.iterrows()):
                pts = v.contours
                ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=color,label=df_.Clusters.values[0])
        ax.legend( loc="lower right", title="Clusters", framealpha=1)
        ax.set_title(f"Cluster {c} and {c2} at Time Frame {t}")
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
    ipyw.interact(update,t=(0,int(df.Time.max()/2)),c=df.Clusters.unique(),c2=df.Clusters.unique());
    
########################################################################################################
    
def plot_matching(df):
    """Compute and plot the match between a baseline and the clustering. NB: The df has to have the ground truth, meaning loading
    must have the add_assigment flag to truth.
    INPUT: Cluster df with assigment
    OUTPUT: None"""
    fig = plt.figure(figsize=(9,9))
    ax = plt.axes( projection='3d')
    
    # Get matching
    
    # find the mapping to the original clusters to our clusters by taking in all points in one of our clusters
    # the most common cluster in the original.
    mapping = df.groupby("Clusters")["Assignment"].apply(lambda x : (Counter(x).most_common(1)[0][0],Counter(x).most_common(1)[0][1],len(x)))
    match = pd.DataFrame(np.stack(mapping.values),columns=["Assignment","N_Recognized","N_Tot"]).set_index("Assignment")
    match = match.drop(-999)
    sum_match = match.groupby(match.index).agg(sum)
    ratio = sum_match["N_Recognized"]/sum_match["N_Tot"]
    match_dict = dict(zip(ratio.index,ratio.values))

    # Here we only want to plot the neurons that are in the clusters we decided to keep
    denoised = df.loc[(df['Assignment'] != -999) & (df['Assignment'] != -1)] 
    pts = np.stack(denoised["centroids"].values)
    noise = df.loc[df['Assignment'] == -999] # show what is original assumed to be noise in white
    # TODO add the other type of noise
    pts_noise = np.stack(noise["centroids"].values)

    colors  = ["tab:blue","tab:orange","tab:green","tab:purple","tab:pink","tab:cyan","tab:red","tab:olive"]


    # plot
    for i,(k,v) in enumerate(denoised.groupby("Clusternames")):
        pts = np.stack(v["centroids"].values)
        match = match_dict[v.Assignment.values[0]]
        scatter = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors[i],label=f"{k}, match at :" + "%.2f" % match + '%')
    scatter = ax.scatter(pts_noise[:,0], pts_noise[:,1], pts_noise[:,2], c='w',alpha=0.09)
    scatter = ax.scatter(pts_noise[0,0], pts_noise[0,1], pts_noise[0,2], c='w',alpha=1,label = "Untracked by Baseline")
    ax.legend(loc="lower left")
    ax.set_title('Matching to Baseline',fontsize=20)
    ax.set_xlabel('$X$')
    ax.set_ylabel('$Y$')
    ax.set_zlabel('$Z$')
    
import os 
from hyperbolic_clustering.hyperbolic_kmeans.hkmeans import HyperbolicKMeans
from hyperbolic_clustering.utils.utils import poincare_dist, poincare_distances
from utils.access_embeddings_utils import get_subnetworks, get_subnetworks_left_right
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.constants_utils import NUM_ROIS, NUM_SUBNETS, NUM_SBJS

def fit_clusters(embeddings_df, num_updates=10, is_tree=False):
    emb_coords = np.array(embeddings_df[['x', 'y']])
    hkmeans = HyperbolicKMeans(n_clusters = embeddings_df.label.nunique())
    hkmeans.n_samples = emb_coords.shape[0]
    hkmeans.init_centroids(radius=0.1)
    hkmeans.init_assign(embeddings_df.label.values)
    for i in range(num_updates):
        hkmeans.update_centroids(emb_coords)
    return {'model': hkmeans, 'embedding': embeddings_df}

def distance_between(A, B, metric='average'):
    # methods for intercluster distances
    distances = []
    for a in A:
        distances += [poincare_dist(a, b) for b in B]
    if metric == 'average':
        return np.mean(distances)
    elif metric == 'max':
        return np.max(distances)
    elif metric == 'min':
        return np.min(distances)
    else:
        print('Invalid metric specified')
        return
        
def distance_within(A, centroid, metric='variance'):
    # methods to compute cohesion within cluster
    centroid_distances = np.array([poincare_dist(x, centroid) for x in A])
    pairwise_distances = poincare_distances(A)
    if metric == 'variance':
        return np.mean(centroid_distances ** 2)
    elif metric == 'diameter':
        return np.max(pairwise_distances)
    elif metric == 'pairwise':
        return np.sum(pairwise_distances) / len(A)
    else:
        print('Invalid metric specified')
        return

def cluster_features(embeddings_df, centroids, wc_metric='pairwise', bc_metric='average'):
    emb_coords = np.array(embeddings_df[['x', 'y']])
    within_cluster = []
    between_cluster = []
    for i in range(len(np.unique(embeddings_df.label))):
        within_cluster.append(distance_within(emb_coords[embeddings_df.label == i], centroid=centroids[i], metric=wc_metric))
        for j in range(i + 1, len(np.unique(embeddings_df.label))):
            between_cluster.append(distance_between(emb_coords[embeddings_df.label == i], emb_coords[embeddings_df.label == j], metric=bc_metric))
    return {'within': np.array(within_cluster), 'between': np.array(between_cluster)}


def get_cortices_to_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices):
    embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
    embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
    cortices_to_hyperbolic_radii_left = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_right = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices)
    return cortices_to_hyperbolic_radii_left, cortices_to_hyperbolic_radii_right

def get_cortices_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    origin = np.array([0, 0])
    cortices_to_hyperbolic_radii = dict()
    for index, centroid in enumerate(hkmeans.centroids):
        hyperbolic_radius = poincare_dist(centroid, origin)
        cortices_to_hyperbolic_radii[cortex_ids_to_cortices[embeddings_df.label.unique()[index]]] = hyperbolic_radius
    return cortices_to_hyperbolic_radii

def get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices):
    origin = np.array([0, 0])
    cortices_to_hyperbolic_radii = dict()
    avg_radius = 0
    for cortex_id in embeddings_df.label.unique():
        embeddings_from_id = embeddings_df[embeddings_df.label == cortex_id][['x', 'y']].values
        avg_radius = 0
        for embedding_coords in embeddings_from_id:
            hyperbolic_radius = poincare_dist(embedding_coords, origin)
            avg_radius += hyperbolic_radius
        avg_radius /= len(embeddings_from_id)
        cortices_to_hyperbolic_radii[cortex_ids_to_cortices[cortex_id]] = avg_radius
    return cortices_to_hyperbolic_radii

def get_roi_hyperbolic_radii_list(embeddings_df):
    """
    Returns a NUM_ROIS-long vector of hyperbolic radii for each ROI in the embeddings_df, a single subject's embeddings data.
    """
    origin = np.array([0, 0])
    roi_hyperbolic_radii = []
    for roi_id in embeddings_df.id.unique():
        embeddings_from_id = embeddings_df[embeddings_df.id == roi_id][['x', 'y']].values
        for embedding_coords in embeddings_from_id:
            hyperbolic_radius = poincare_dist(embedding_coords, origin)
            roi_hyperbolic_radii.append(hyperbolic_radius)

    return roi_hyperbolic_radii

def get_roi_hyperbolic_radii_vector_left_right(embeddings_df):
    """
    Returns a two (NUM_ROIS // 2)-long vectors (Left, Right) of hyperbolic radii for each ROI in the embeddings_df, a single subject's embeddings data.
    """
    origin = np.array([0, 0])
    roi_hyperbolic_radii_L = []
    roi_hyperbolic_radii_R = []
    for roi_id in embeddings_df.id.unique():
        embeddings_from_id_L = embeddings_df[(embeddings_df.id == roi_id) & (embeddings_df.LR == 'L')][['x', 'y']].values
        embeddings_from_id_R = embeddings_df[(embeddings_df.id == roi_id) & (embeddings_df.LR == 'R')][['x', 'y']].values
        for embedding_coords_L in embeddings_from_id_L:
            hyperbolic_radius_L = poincare_dist(embedding_coords_L, origin)
            roi_hyperbolic_radii_L.append(hyperbolic_radius_L)
        for embedding_coords_R in embeddings_from_id_R:
            hyperbolic_radius_R = poincare_dist(embedding_coords_R, origin)
            roi_hyperbolic_radii_R.append(hyperbolic_radius_R)
    
    return roi_hyperbolic_radii_L, roi_hyperbolic_radii_R 

def get_cortex_regions_to_hyperbolic_radii_across_age_left_right(embeddings_df_list_with_age_labels, cortex_ids_to_cortices):
    """
    22 x 587 (NUM_SUBNETS x NUM_SBJS)
    """
    cortices_to_hyperbolic_radii_across_age_L = dict()
    cortices_to_hyperbolic_radii_across_age_R = dict()
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))

    for embeddings_df in embeddings_df_list_with_age_labels:
        for cortex_id in range(1, NUM_SUBNETS + 1):
            embeddings_from_id_L = embeddings_df.loc[(embeddings_df.label == cortex_id) & (embeddings_df.LR == "L"), ['x', 'y']].values
            embeddings_from_id_R = embeddings_df.loc[(embeddings_df.label == cortex_id) & (embeddings_df.LR == "R"), ['x', 'y']].values
            
            avg_cortex_hyp_radius_L = get_poincare_avg_radius(embeddings_from_id_L)
            avg_cortex_hyp_radius_R = get_poincare_avg_radius(embeddings_from_id_R)
            
            cortex = cortex_ids_to_cortices[cortex_id]
            cortices_to_hyperbolic_radii_across_age_L[cortex] = cortices_to_hyperbolic_radii_across_age_L.get(cortex, []) + [avg_cortex_hyp_radius_L]
            cortices_to_hyperbolic_radii_across_age_R[cortex] = cortices_to_hyperbolic_radii_across_age_R.get(cortex, []) + [avg_cortex_hyp_radius_R]
            
    return cortices_to_hyperbolic_radii_across_age_L, cortices_to_hyperbolic_radii_across_age_R

def get_subnetwork_hyperbolic_radii_per_sbj_left_right(date, precalculated_radii=None):
    """
    587 x 22 (NUM_SBJS x NUM_SUBNETS) Dictionary
    TODO: Implement prioritizing subject order
    """
    # MIGHT LEAD TO CYCLE IN IMPORTS TODO: CHECK FOR BUGS
    from visualization import get_average_roi_hyperbolic_radii_per_sbj_across_runs
    subnetwork_hyperbolic_radii_per_sbj_L = [dict() for sbj_num in range(NUM_SBJS)]
    subnetwork_hyperbolic_radii_per_sbj_R = [dict() for sbj_num in range(NUM_SBJS)]
    if not precalculated_radii:
        radii_per_sbj_per_roi = get_average_roi_hyperbolic_radii_per_sbj_across_runs(date)
    else:
        radii_per_sbj_per_roi = precalculated_radii
        if type(radii_per_sbj_per_roi) != dict: raise AssertionError(f"Invalid precalculated_radii type! {type(precalculated_radii)}")
    for sbj_num in range(NUM_SBJS):
        for cortex_index in range(NUM_SUBNETS):
            subnetworks_L, subnetworks_R = get_subnetworks_left_right() # 0-indexed

            # TODO: CHECK THIS IS CORRECT
            subnetwork_radii_L = [radii_per_sbj_per_roi[sbj_num][index] for index in subnetworks_L[cortex_index]]
            subnetwork_radii_R = [radii_per_sbj_per_roi[sbj_num][index] for index in subnetworks_R[cortex_index]]
            # Make sure is by cortex index or by cortex string
            subnetwork_hyperbolic_radii_per_sbj_L[sbj_num][cortex_index] = sum(subnetwork_radii_L) / len(subnetwork_radii_L) 
            subnetwork_hyperbolic_radii_per_sbj_R[sbj_num][cortex_index] = sum(subnetwork_radii_R) / len(subnetwork_radii_R) 
    return subnetwork_hyperbolic_radii_per_sbj_L, subnetwork_hyperbolic_radii_per_sbj_R

def get_age_labels_to_thickness(roi_index):
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    thicks_myelins_tensor = np.load(os.path.join("data", "cam_can_multiple", "cam_can_thicks_myelins_tensor_592_filtered.npy"))
    THICK_INDEX = 0
    age_labels_to_thickness = dict()
    thicks = [thicks_myelins_tensor[THICK_INDEX][graph_index][roi_index] for graph_index in range(587)]
    for age_label, thick in zip(age_labels, thicks):
        age_labels_to_thickness[age_label] = age_labels_to_thickness.get(age_label, []) + [thick]
    return age_labels_to_thickness

def get_age_labels_to_myelination(roi_index):
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    thicks_myelins_tensor = np.load(os.path.join("data", "cam_can_multiple", "cam_can_thicks_myelins_tensor_592_filtered.npy"))
    MYELIN_INDEX = 1
    age_labels_to_myels = dict()
    myelins = [thicks_myelins_tensor[MYELIN_INDEX][graph_index][roi_index] for graph_index in range(587)]
    for age_label, myelin in zip(age_labels, myelins):
        age_labels_to_myels[age_label] = age_labels_to_myels.get(age_label, []) + [myelin]
    return age_labels_to_myels

def get_age_labels_to_total_edges_by_threshold(threshold):
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    plv_tensor = np.load(os.path.join("data", "cam_can_multiple", "plv_tensor_592_sbj_filtered.npy"))
    for sbj_index in range(len(plv_tensor)):
        for i in range(NUM_ROIS):
            plv_tensor[sbj_index][i][i] = 0
    
    plv_matrices = []
    for sbj_index in range(len(plv_tensor)):
        plv_matrix = plv_tensor[sbj_index].copy()
        plv_matrix [plv_matrix >= threshold] = 1
        plv_matrix [plv_matrix < threshold] = 0
        plv_matrices.append(plv_matrix)
        
    age_labels_to_total_edges = dict()
    for age_label, plv_matrix in zip(age_labels, plv_matrices):
        age_labels_to_total_edges[age_label] = age_labels_to_total_edges.get(age_label, []) + [np.sum(plv_matrix)]
    return age_labels_to_total_edges

def get_cortices_to_avg_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices):
    
    cortices_to_hyperbolic_radii_L = dict()
    cortices_to_hyperbolic_radii_R = dict()
    for cortex_id in embeddings_df.label.unique():
        
        embeddings_from_id_L = embeddings_df.loc[(embeddings_df.label == cortex_id) & (embeddings_df.LR == "L"), ['x', 'y']].values
        embeddings_from_id_R = embeddings_df.loc[(embeddings_df.label == cortex_id) & (embeddings_df.LR == "R"), ['x', 'y']].values

        cortices_to_hyperbolic_radii_L[cortex_ids_to_cortices[cortex_id]] = get_poincare_avg_radius(embeddings_from_id_L)
        cortices_to_hyperbolic_radii_R[cortex_ids_to_cortices[cortex_id]] = get_poincare_avg_radius(embeddings_from_id_R)
    
    return cortices_to_hyperbolic_radii_L, cortices_to_hyperbolic_radii_R 

def get_poincare_avg_radius(embeddings_from_id):    
    origin = np.array([0, 0])
    avg_radius = 0
    for embedding_coords in embeddings_from_id:
        hyperbolic_radius = poincare_dist(embedding_coords, origin)
        avg_radius += hyperbolic_radius
    avg_radius /= len(embeddings_from_id)
    return avg_radius


def get_depths_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering=None, is_tree=False):
    if not clustering: clustering = fit_clusters(embeddings_df, is_tree=is_tree)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    origin = np.array([0, 0])
    depths_to_hyperbolic_radii = dict()
    for index, centroid in enumerate(hkmeans.centroids):
        hyperbolic_radius = poincare_dist(centroid, origin)
        depths_to_hyperbolic_radii[index] = hyperbolic_radius
    return depths_to_hyperbolic_radii

def get_depths_to_avg_hyperbolic_cluster_radii(embeddings_df):
    origin = np.array([0, 0])
    depths_to_avg_hyperbolic_radii = dict()
    avg_radius = 0
    for depth in embeddings_df.label.unique():
        embeddings_from_depth = embeddings_df[embeddings_df.label == depth][['x', 'y']].values
        avg_radius = 0
        for embedding_coords in embeddings_from_depth:
            hyperbolic_radius = poincare_dist(embedding_coords, origin)
            avg_radius += hyperbolic_radius
        avg_radius /= len(embeddings_from_depth)
        depths_to_avg_hyperbolic_radii[depth] = avg_radius
    return depths_to_avg_hyperbolic_radii

def get_cortices_to_hyperbolic_cluster_cohesion(embeddings_df, cortex_ids_to_cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    cohesions = cluster_features(embeddings_df, hkmeans.centroids)['within']
    cortices_to_hyperbolic_cohesions = dict()
    for index, cohesion in enumerate(cohesions):
        cortices_to_hyperbolic_cohesions[cortex_ids_to_cortices[embeddings_df.label.unique()[index]]] = cohesion
    return cortices_to_hyperbolic_cohesions

def get_hyperbolic_cluster_metrics(embeddings_df, cortices, clustering=None):
    hyp_cluster_metrics = dict()
    within_cluster_features = []

    # NOTE: Can be adapted into for loop to iterate over all subject embeddings...
    if not clustering: clustering = fit_clusters(embeddings_df)
    embeddings_df = clustering['embedding']
    hkmeans = clustering['model']
    within_cluster_features.append(cluster_features(embeddings_df, hkmeans.centroids)['within'])

    eval_cluster = pd.DataFrame(np.ravel(np.array(within_cluster_features)), columns=['within_cluster_cohesion'])
    # TODO: Can use to give Age Labels!
    graph_labels = np.array([0])
    eval_cluster['label'] = np.repeat(graph_labels, NUM_SUBNETS)
    eval_cluster['label'] = eval_cluster.label.apply(lambda x: ['healthy', 'diagnosed'][int(x)])
    eval_cluster['network'] = np.tile(cortices.unique(), 1)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="network", y="within_cluster_cohesion", hue="label", data=eval_cluster, palette="pastel")
    plt.title('Intra-Cluster Analysis: Brain Networks', size=16)
    plt.show(); 

def plot_between_cluster_metrics(embeddings_df, cortices, clustering=None):
    if not clustering: clustering = fit_clusters(embeddings_df)
    hkmeans = clustering['model']
    embeddings_df = clustering['embedding']
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    # NUM_SUBNETS = cortices.nunique()
    subnet_dist = np.zeros((NUM_SUBNETS, NUM_SUBNETS))
    for i in range(NUM_SUBNETS):
        for j in range(i + 1, NUM_SUBNETS):
            subnet_dist[i, j] = cluster_features(embeddings_df, hkmeans.centroids)['between'][i + j - 1]
            subnet_dist[j, i] = subnet_dist[i, j]

    sns.heatmap(subnet_dist, annot=True);
    ax = plt.gca()
    
    
    ax.set_xticklabels(cortices.unique())
    ax.set_yticklabels(cortices.unique())
    plt.title('Cam_Can_Average_276')
    plt.suptitle('Hyperbolic Distances: Between Network Clusters', size=16)
    plt.show();
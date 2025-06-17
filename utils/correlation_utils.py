import os
from typing import List
import numpy as np
from utils.data_utils import get_myelin_features, get_thick_features, min_max_normalize
from hyperbolic_clustering.hyperbolic_cluster_metrics import get_roi_hyperbolic_radii_list
from utils.access_embeddings_utils import get_embeddings_df, get_embeddings_df_list, get_embeddings_df_list_by_decade, \
                                            get_embeddings_df_list_with_age_labels, get_meg_embeddings_df, get_tree_embeddings_df, \
                                            get_embeddings_df_list_all_splits, get_cortices_and_cortex_ids_to_cortices, scale_embeddings_df_to_poincare_disk, \
                                            get_hcp_atlas_df, get_subnetworks

from utils.constants_utils import NUM_SUBNETS, NUM_ROIS
import networkx as nx
import matplotlib.pyplot as plt
from utils.constants_utils import NUM_SBJS, FIVE_PERCENT_THRESHOLD
from scipy.stats import spearmanr
from visualization import get_average_roi_hyperbolic_radii_per_sbj_across_runs

def correlate_average_hyperbolic_radius_to_thickness(date, subject_index):
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    thick_features = get_myelin_features(subject_index, data_path)
    hyperbolic_radii_vectors = [get_roi_hyperbolic_radii_list(embeddings_df) for embeddings_df in embeddings_df_list ]
    hyperbolic_radii_distribution_per_roi = np.array(hyperbolic_radii_vectors)   
    
    average_hyperbolic_radii = np.sum(hyperbolic_radii_distribution_per_roi, axis=0)

    min_max_normalize_thick_features = min_max_normalize(thick_features)
    min_max_normalize_average_hyperbolic_radii = min_max_normalize(average_hyperbolic_radii)
    return np.corrcoef(min_max_normalize_average_hyperbolic_radii, min_max_normalize_thick_features)[0, 1]

def correlate_hyperbolic_radii_to_myelination(embeddings_df, subject_index):
    hyperbolic_radii = get_roi_hyperbolic_radii_list(embeddings_df)
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    # Seems like might have to min-max normalize thickness features since they vary from 2 - 3, 
    # while hyperbolic radii varies from 0 to 3, also should normalize
    myelin_features = get_myelin_features(subject_index, data_path)
    min_max_normalize_myelin_features = min_max_normalize(myelin_features)
    min_max_normalize_hyperbolic_radii = min_max_normalize(hyperbolic_radii)
    # return np.corrcoef(hyperbolic_radii, thick_features)[0, 1]
    return np.corrcoef(min_max_normalize_hyperbolic_radii, min_max_normalize_myelin_features)[0, 1]


def correlate_difference_in_hyperbolic_radii_to_thickness(embeddings_df_thickness, embeddings_df_identity, subject_index):
    hyperbolic_radii_thickness = get_roi_hyperbolic_radii_list(embeddings_df_thickness)
    hyperbolic_radii_identity = get_roi_hyperbolic_radii_list(embeddings_df_identity)
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    # Seems like might have to min-max normalize thickness features since they vary from 2 - 3, 
    # while hyperbolic radii varies from 0 to 3, also should normalize
    thick_features = get_thick_features(subject_index, data_path)
    from utils.data_utils import min_max_normalize
    min_max_normalize_thick_features = min_max_normalize(thick_features)
    min_max_normalize_hyperbolic_radii_thickness = min_max_normalize(hyperbolic_radii_thickness)
    min_max_normalize_hyperbolic_radii_identity = min_max_normalize(hyperbolic_radii_identity)
    # return np.corrcoef(hyperbolic_radii, thick_features)[0, 1]
    return np.corrcoef(abs(min_max_normalize_hyperbolic_radii_thickness - min_max_normalize_hyperbolic_radii_identity), 
                        min_max_normalize_thick_features)[0, 1]

def correlate_difference_in_hyperbolic_radii_to_myelination(embeddings_df_myelination, embeddings_df_identity, subject_index):
    hyperbolic_radii_myelination = get_roi_hyperbolic_radii_list(embeddings_df_myelination)
    hyperbolic_radii_identity = get_roi_hyperbolic_radii_list(embeddings_df_identity)
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    # Seems like might have to min-max normalize thickness features since they vary from 2 - 3, 
    # while hyperbolic radii varies from 0 to 3, also should normalize
    myelin_features = get_myelin_features(subject_index, data_path)
    from utils.data_utils import min_max_normalize
    min_max_normalize_myelin_features = min_max_normalize(myelin_features)
    min_max_normalize_hyperbolic_radii_myelination = min_max_normalize(hyperbolic_radii_myelination)
    min_max_normalize_hyperbolic_radii_identity = min_max_normalize(hyperbolic_radii_identity)
    # return np.corrcoef(hyperbolic_radii, thick_features)[0, 1]
    return np.corrcoef(abs(min_max_normalize_hyperbolic_radii_myelination - min_max_normalize_hyperbolic_radii_identity), 
                        min_max_normalize_myelin_features)[0, 1]


def correlate_hyperbolic_radii_to_thickness(embeddings_df, subject_index):
    hyperbolic_radii = get_roi_hyperbolic_radii_list(embeddings_df)
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    # Seems like might have to min-max normalize thickness features since they vary from 2 - 3, 
    # while hyperbolic radii varies from 0 to 3, also should normalize
    thick_features = get_thick_features(subject_index, data_path)
    from utils.data_utils import min_max_normalize
    min_max_normalize_thick_features = min_max_normalize(thick_features)
    min_max_normalize_hyperbolic_radii = min_max_normalize(hyperbolic_radii)
    # return np.corrcoef(hyperbolic_radii, thick_features)[0, 1]
    return np.corrcoef(min_max_normalize_hyperbolic_radii, min_max_normalize_thick_features)[0, 1]

# TODO: Add parameter thick/myelin to simplify methods
def consolidate_correlations_fisher_z(date, thickness_or_myelination="thickness"):
    if thickness_or_myelination == "thickness":
        correlations = get_correlations_hyperbolic_radii_to_thickness(date)
    elif thickness_or_myelination == "myelination":
        correlations = get_correlations_hyperbolic_radii_to_myelination(date)
    else:
        raise ValueError("Invalid value for thickness_or_myelination: {}".format(thickness_or_myelination))
    # Step 1: Apply Fisher's Z-transform to each correlation coefficient
    transformed_correlations = np.arctanh(correlations)
    
    # Step 2: Average the transformed correlation coefficients
    consolidated_transformed_correlation = np.mean(transformed_correlations)
    
    # Step 3: Apply inverse Fisher's Z-transform to obtain the consolidated correlation coefficient
    consolidated_correlation = np.tanh(consolidated_transformed_correlation)
    
    return consolidated_correlation

def get_correlations_hyperbolic_radii_to_thickness(date):
    SBJ_INDEX = 100
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    correlations = [correlate_hyperbolic_radii_to_thickness(embeddings_df, SBJ_INDEX) for embeddings_df in embeddings_df_list]
    return correlations

def get_correlations_hyperbolic_radii_to_myelination(date):
    SBJ_INDEX = 100
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!

    correlations = [correlate_hyperbolic_radii_to_myelination(embeddings_df, SBJ_INDEX) for embeddings_df in embeddings_df_list]
    return correlations


def get_difference_correlations_hyperbolic_radii_to_thickness(date):
    SBJ_INDEX = 100
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    embeddings_df_list_identity = get_embeddings_df_list("2023_7_13")
    correlations = [correlate_difference_in_hyperbolic_radii_to_thickness(embeddings_df, embeddings_df_id, SBJ_INDEX) 
                    for embeddings_df, embeddings_df_id in zip(embeddings_df_list, embeddings_df_list_identity)]
    return correlations

def get_difference_correlations_hyperbolic_radii_to_myelination(date):
    SBJ_INDEX = 100
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    embeddings_df_list_identity = get_embeddings_df_list("2023_7_13")
    correlations = [correlate_difference_in_hyperbolic_radii_to_myelination(embeddings_df, embeddings_df_id, SBJ_INDEX) 
                    for embeddings_df, embeddings_df_id in zip(embeddings_df_list, embeddings_df_list_identity)]
    return correlations

# TODO: Add parameter thick/myelin to simplify methods
def consolidate_difference_correlations_fisher_z(date, thickness_or_myelination="thickness"):
    if thickness_or_myelination == "thickness":
        correlations = get_difference_correlations_hyperbolic_radii_to_thickness(date)
    elif thickness_or_myelination == "myelination":
        correlations = get_difference_correlations_hyperbolic_radii_to_myelination(date)
    else:
        raise ValueError("Invalid value for thickness_or_myelination: {}".format(thickness_or_myelination))
    # Step 1: Apply Fisher's Z-transform to each correlation coefficient
    transformed_correlations = np.arctanh(correlations)
    
    # Step 2: Average the transformed correlation coefficients
    consolidated_transformed_correlation = np.mean(transformed_correlations)
    
    # Step 3: Apply inverse Fisher's Z-transform to obtain the consolidated correlation coefficient
    consolidated_correlation = np.tanh(consolidated_transformed_correlation)
    
    return consolidated_correlation

def get_plv_matrices():
    """
    PLV Binarized Matrices using 5 % Edge Threshold
    """
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    plv_tensor = np.load(os.path.join("data", "cam_can_multiple", "plv_tensor_592_sbj_filtered.npy"))
    NUM_ROIS = 360
    for sbj_index in range(len(plv_tensor)):
        for i in range(NUM_ROIS):
            plv_tensor[sbj_index][i][i] = 0
    
    plv_matrices = []
    for sbj_index in range(len(plv_tensor)):
        plv_matrix = plv_tensor[sbj_index].copy()
        plv_matrix [plv_matrix >= FIVE_PERCENT_THRESHOLD] = 1
        plv_matrix [plv_matrix < FIVE_PERCENT_THRESHOLD] = 0
        plv_matrices.append(plv_matrix)
    return plv_matrices
def plot_graph_level_measures():
    names_to_measures = dict()
    plv_matrices = get_plv_matrices()
    subnetworks = get_subnetworks()
    for index, graph_theoretic_measure in enumerate([nx.average_shortest_path_length,
                                    nx.global_efficiency,
                                    nx.average_clustering,
                                    nx.transitivity,
                                    nx.local_efficiency,
                                    nx.community.modularity,
                                    nx.closeness_centrality,
                                    nx.katz_centrality_numpy,
                                    nx.betweenness_centrality,
                                    nx.degree_assortativity_coefficient,
                                    nx.average_neighbor_degree,
                                    nx.average_degree_connectivity]):
        measures = []
        for sbj_index in range(NUM_SBJS):
            G = nx.from_numpy_matrix(plv_matrices[sbj_index])
            if graph_theoretic_measure == nx.community.modularity:
                measure = graph_theoretic_measure(G, communities=subnetworks)
            elif graph_theoretic_measure == nx.average_shortest_path_length:
                try :
                    avg_len = nx.average_shortest_path_length(G)
                    measure = avg_len
                except: 
                    avg_len = num_components = 0
                    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
                        avg_len += nx.average_shortest_path_length(C)
                        num_components += 1
                    avg_len /= num_components
                    measure = avg_len
            else:         
                measure = graph_theoretic_measure(G)
            if type(measure) == dict:
                measure = sum(measure.values()) / len(measure.values())
            measures.append(measure)
        plt.figure()
        plt.xlabel("Subject Number")
        y_strs = ["average_shortest_path_length", "global_efficiency", 
                "average_clustering_coefficient", "transitivity", 
                "local_efficiency", "modularity", "closeness_centrality", "katz_centrality", 
                "betweenness_centrality", "degree_assortativity_coefficient", 
                "average_neighbor_degree", "average_degree_connectivity"]
        y_str = y_strs[index]
        capitalized = [cap_str[0].upper() + cap_str[1 : ] + " " for cap_str in \
                        y_strs[index].replace("_", " ").split()]
        title_str = "".join(capitalized)[ : -1]
        plt.ylabel(y_str)
        plt.title(title_str)
        plt.plot(measures)
        names_to_measures[title_str] = measures
    return names_to_measures


def get_node_level_measure_spearman_correlation_across_age(date: str, measure_str: str, precalculated_radii : List = None):
    """
    Return average of 587 calculated spearman correlations
    Return average of 587 calculated p-values
    Return 587 x 360 matrix of graph measure values (NUM_SBJS x NUM_ROIS)
    """
    if measure_str == "clustering": measure = nx.clustering
    elif measure_str == "degree_centrality": measure = nx.degree_centrality
    elif measure_str == "katz_centrality": measure = nx.katz_centrality_numpy
    elif measure_str == "closeness_centrality": measure = nx.closeness_centrality
    elif measure_str == "betweenness_centrality": measure = nx.betweenness_centrality
    elif measure_str == "shortest_path_lengths": measure = nx.all_pairs_shortest_path_length
    elif measure_str == "average_neighbor_degree": measure = nx.average_neighbor_degree
    elif measure_str == "efficiency": measure = nx.efficiency
    else: raise AssertionError(f"Invalid graph measure type: {measure_str} !")

    measure_title_str = "Clustering Coefficient" if measure_str == "clustering" \
        else "Degree Centrality" if measure_str == "degree_centrality" \
        else "Katz Centrality" if measure_str == "katz_centrality" \
        else "Closeness Centrality" if measure_str == "closeness_centrality" \
        else "Betweenness Centrality" if measure_str == "betweenness_centrality" \
        else "Shortest Path Lengths" if measure_str == "shortest_path_lengths" \
        else "Average Neighbor Degree" if measure_str == "average_neighbor_degree" \
        else "Efficiency" if measure == "efficiency" \
        else "Invalid Graph Measure Type"
    plv_matrices = get_plv_matrices()

    # TODO: Fix Embeddings DF method of capturing all subject data!!!!!

    # CHANGE SO THAT THIS RESPECTS ORDERING OF AGE LABELS
    if not precalculated_radii:
        average_roi_hyperbolic_radii_list_per_sbj = get_average_roi_hyperbolic_radii_per_sbj_across_runs(date)
    else:
        average_roi_hyperbolic_radii_list_per_sbj = precalculated_radii
        if type(average_roi_hyperbolic_radii_list_per_sbj) != dict: raise AssertionError(f"Invalid Precalculated Radius data type! {precalculated_radii}")
    spearman_correlations = []
    p_values = []
    graph_measure_587_360_nodes = dict()
    print(f"Calculating Spearman Correlation for Graph Measure: {measure_str}:")
    for sbj_index in range(NUM_SBJS):
        print("Calculating Spearman Correlation for Subject Index: ", sbj_index, "...")
        sbj_360_roi_radii = average_roi_hyperbolic_radii_list_per_sbj[sbj_index]
        G_sbj = nx.from_numpy_matrix(plv_matrices[sbj_index])
        if measure_str == "shortest_path_lengths":
            graph_measure_360_nodes = {node_1: dict() for node_1 in range(360)}
            shortest_lens = [*nx.all_pairs_shortest_path_length(G_sbj)]
            for node_1 in range(NUM_ROIS):
                for node_2 in range(NUM_ROIS):
                    try:
                        graph_measure_360_nodes[node_1][node_2] = shortest_lens[node_1][1][node_2] 
                    except: # TODO: Careful with unconnected components
                        graph_measure_360_nodes[node_1][node_2] = NUM_ROIS # Path is not connected
            for node_1 in range(NUM_ROIS):
                graph_measure_360_nodes[node_1] = sum(np.array([graph_measure_360_nodes[node_1][node_2] for node_2 in range(NUM_ROIS)])) / NUM_ROIS
        elif measure_str == "efficiency":
            
            graph_measure_360_nodes = {node_1: dict() for node_1 in range(360)}
            for node_1 in range(NUM_ROIS):
                for node_2 in range(NUM_ROIS):
                    try:
                        efficiency = nx.efficiency(G_sbj, node_1, node_2)
                    except:
                        efficiency = 2
                    graph_measure_360_nodes[node_1][node_2] = efficiency
            for node_1 in range(NUM_ROIS):
                graph_measure_360_nodes[node_1] = sum(np.array([graph_measure_360_nodes[node_1][node_2] for node_2 in range(NUM_ROIS)])) / NUM_ROIS
        else:
            graph_measure_360_nodes = measure(G_sbj)

        measures = [*graph_measure_360_nodes.values()]        
        spearman_correlation, p_value = spearmanr(sbj_360_roi_radii, measures)
        
        graph_measure_587_360_nodes[sbj_index] = measures
        p_values.append(p_value)
        spearman_correlations.append(spearman_correlation)

    plt.figure()
    plt.title(f"Spearman Correlation for {measure_title_str} and Hyperbolic Radii Across Age")
    plt.xlabel("Subject Index")
    plt.ylabel("Spearman Correlation")
    plt.ylim(-1, 1)
    print("THESE ARE THE CORRELATIONS")
    plt.plot(spearman_correlations)

    plt.figure()
    plt.title(f"Spearman Correlation P-Value for {measure_title_str} and Hyperbolic Radii Across Age")
    plt.xlabel("Subject Index")
    plt.ylabel("P-Value")
    significance_threshold = 0.05
    plt.axhline(significance_threshold, color='black', linestyle='--', label='Significance Threshold (0.05)')
    plt.ylim(0, 1)
    plt.plot(p_values)
    # Only scatter plot significant p-values below 0.05
    for x_index, p_value in enumerate(p_values):
        if p_value < significance_threshold: 
            plt.scatter(x_index, p_value, marker='v', color='green')

    return sum(spearman_correlations) / len(spearman_correlations), sum(p_values) / len(p_values), graph_measure_587_360_nodes
    # for each measure:

        # 360 ROI node measure vector per subject LEFT / RIGHT

        # 587 Spearman Correlations plotted LEFT / RIGHTs


import os 
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
from matplotlib.patches import Circle 
from matplotlib.colors import LinearSegmentedColormap
from hyperbolic_clustering.hyperbolic_cluster_metrics import get_hyperbolic_cluster_metrics, \
                    plot_between_cluster_metrics, get_cortices_to_hyperbolic_cluster_radii, \
                    get_cortices_to_hyperbolic_cluster_cohesion, get_cortices_to_hyperbolic_cluster_radii_left_right, \
                    get_cortices_to_avg_hyperbolic_cluster_radii, get_depths_to_avg_hyperbolic_cluster_radii, \
                    fit_clusters, get_depths_to_hyperbolic_cluster_radii, get_cortices_to_avg_hyperbolic_cluster_radii_left_right, \
                    get_roi_hyperbolic_radii_list, get_age_labels_to_total_edges_by_threshold, get_cortex_regions_to_hyperbolic_radii_across_age_left_right

from utils.data_utils import get_thick_features, get_myelin_features, min_max_normalize
from utils.access_embeddings_utils import get_embeddings_df, get_embeddings_df_list, get_embeddings_df_list_by_decade, \
                                            get_embeddings_df_list_with_age_labels, get_meg_embeddings_df, get_tree_embeddings_df, \
                                            get_embeddings_df_list_all_splits, get_cortices_and_cortex_ids_to_cortices, scale_embeddings_df_to_poincare_disk, \
                                            get_hcp_atlas_df
import matplotlib.colors as mcolors
from utils.constants_utils import ROI_NAMES_360, FIVE_PERCENT_THRESHOLD, SIX_PERCENT_THRESHOLD, SEVEN_PERCENT_THRESHOLD, EIGHT_PERCENT_THRESHOLD, \
                                NUM_DECADES, DECADE_POSITIONS_STR_LIST, THRESHOLDS_TO_EDGE_FRACTIONS, DATAPATH, ROI_DIST, CORTEX_IDS_TO_ROI_DIST, \
                                NUM_ROIS, CORTEX_TO_ABBREVIATION, NUM_SBJS, NUM_SUBNETS, COLORS, CAMCAN_DATSET_STR
from utils.threshold_utils import get_age_prediction_metrics_with_thresholds
os.environ['DATAPATH'] = DATAPATH
os.environ['LOG_DIR'] = os.path.join(os.getcwd(), 'logs')

# MEG_COLORS = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', 
#             '#00ffff', '#800000', '#008000', '#000080', '#808000', 
#             '#800080', '#008080', '#ff8080', '#80ff80', '#8080ff', 
#             '#ffff80', '#ff80ff', '#80ffff', '#c00000', '#00c000', 
#             '#0000c0', '#c0c000', '#c000c0']

def viz_metrics(log_path, use_save=False, prefix= ""):
    def collect_and_plot_curvatures(log_path, use_save, prefix):
        curvatures = []
        with open(log_path) as f:
            for line in f.readlines():
                if "INFO:root:Model" in line:
                    words = line.split()
                    curvatures.append(float(words[3]))    
        if not curvatures: return
        plot_values(curvatures, "Model Curvature", use_save, prefix)
    def collect_loss_roc_ap_for_train_val_test_from_log(log_path):
        log_path
        train_losses = []
        train_rocs = []
        train_aps = []
        val_losses = []
        val_rocs = []
        val_aps = []
        test_losses = []
        test_rocs = []
        test_aps = []
        test_epochs = []
        with open(log_path) as f:
            for line in f.readlines():

                if "INFO:root:Epoch:" in line:
                    words = line.split()
                    if "train_loss" in line:
                        train_losses.append(float(words[6]))
                        train_rocs.append(float(words[8]))
                        train_aps.append(float(words[10]))
                    if "val_loss" in line:
                        val_losses.append(float(words[3]))
                        val_rocs.append(float(words[5]))
                        val_aps.append(float(words[7]))
                if "INFO:root:Test Epoch" in line:
                    words = line.split()
                    test_losses.append(float(words[4]))
                    test_rocs.append(float(words[6]))
                    test_aps.append(float(words[8]))
                if "INFO:root:Model Improved;" in line:
                    words = line.split()
                    test_epochs.append(int(words[5]))
                    test_losses.append(float(words[7]))
                    test_rocs.append(float(words[9]))
                    test_aps.append(float(words[11]))
                if "Last Epoch:" in line:
                    words = line.split()
                    test_epochs.append(int(words[2]))
        return train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, test_losses, test_rocs, test_aps, test_epochs
    
    metrics = collect_loss_roc_ap_for_train_val_test_from_log(log_path)
    train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, \
        test_losses, test_rocs, test_aps, test_epochs = metrics
    def plot_values(values, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        epochs = [5 * i for i in range(len(values))] # Model Curvature actually is reported every epoch
        plt.title(title)
        plt.xlabel("Epoch")
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    def plot_values_with_epochs(values, epochs, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.xticks(rotation=45)
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    
    datas = [train_losses,
                val_losses,
                test_losses,
                train_aps,
                val_aps,
                test_aps,
                train_rocs,
                val_rocs,
                test_rocs]
    titles = ["Train_Loss",
            "Validation_Loss",
            "Test_Loss",
            "Train_Average_Precision",
            "Validation_Average_Precision",
            "Test_Average_Precision",
            "Train_ROC",
            "Validation_ROC",
            "Test_ROC"]
    
    for i, ax in enumerate(axes.flat):
        
        data = datas[i]
        title = titles[i]
        # epochs = [5 * i for i in range(len(data))] if i % 3 != 2 else test_epochs
        epochs = [5 * i for i in range(len(data))]
        # Plot the data in the current subplot
        ax.plot(epochs, data)
        ax.set_title(title)

    # Adjust spacing between subplots
    fig.tight_layout()

    print(f"Test_Loss: {test_losses[-1]}")
    print(f"Test_ROC: {test_rocs[-1]}")
    print(f"Test_Average_Precision: {test_aps[-1]}")


def viz_metrics_multiple(log_path, use_save=False, prefix= ""):
    def collect_and_plot_curvatures(log_path, use_save, prefix):
        curvatures = []
        with open(log_path) as f:
            for line in f.readlines():
                if "INFO:root:Model" in line:
                    words = line.split()
                    curvatures.append(float(words[3]))    
        if not curvatures: return
        plot_values(curvatures, "Model Curvature", use_save, prefix)
    def collect_loss_roc_ap_for_train_val_test_from_log_multiple(log_path):
        log_path
        train_losses = []
        train_rocs = []
        train_aps = []
        train_accs = []
        val_losses = []
        val_rocs = []
        val_aps = []
        val_accs = []
        test_losses = []
        test_rocs = []
        test_aps = []
        test_epochs = []
        test_accs = []
        with open(log_path) as f:
            for line in f.readlines():

                if "INFO:root:Epoch:" in line:
                    words = line.split()
                    if "train_loss" in line:
                        train_losses.append(float(words[6]))
                        train_rocs.append(float(words[8]))
                        train_aps.append(float(words[10]))
                        # train_accs.append()
                    if "val_loss" in line:
                        val_losses.append(float(words[3]))
                        val_rocs.append(float(words[5]))
                        val_aps.append(float(words[7]))
                        # val_accs.append()
                if "INFO:root:Val Epoch" in line:
                    words = line.split()
                    # val_losses.append(float(words[12][ : -1]))
                    # val_rocs.append(float(words[10][ : -1]))
                    # val_aps.append(float(words[9][ : -4]))
                    val_accs.append(float(words[15][ : -1]))
                    val_losses.append(float(words[13][ : -1]))
                    val_rocs.append(float(words[11][ : -1]))
                    val_aps.append(float(words[9][ : -1]))
                if "INFO:root:Test Epoch" in line:
                    words = line.split()
                    test_accs.append(float(words[15][ : -1]))
                    test_losses.append(float(words[13][ : -1]))
                    test_rocs.append(float(words[11][ : -1]))
                    test_aps.append(float(words[9][ : -1]))
                if "INFO:root:Model Improved;" in line:
                    words = line.split()
                    test_epochs.append(int(words[5]))
                    test_losses.append(float(words[7]))
                    test_rocs.append(float(words[9]))
                    test_aps.append(float(words[11]))
                    # test_accs.append()
                if "Last Epoch:" in line:
                    words = line.split()
                    test_epochs.append(int(words[2]))
        return train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, test_losses, test_rocs, test_aps, test_accs, test_epochs
    
    metrics = collect_loss_roc_ap_for_train_val_test_from_log_multiple(log_path)
    train_losses, train_rocs, train_aps, val_losses, val_rocs, val_aps, \
        test_losses, test_rocs, test_aps, test_accs, test_epochs = metrics
    def plot_values(values, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        epochs = [5 * i for i in range(len(values))] # Model Curvature actually is reported every epoch
        plt.title(title)
        plt.xlabel("Epoch")
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    def plot_values_with_epochs(values, epochs, title, use_save, prefix=""): 
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel("Epoch")
        plt.xticks(rotation=45)
        plt.plot(epochs, values)
        if use_save: plt.savefig(f"{prefix}_{title}.png" if prefix else f"{title}.png")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 10))
    
    datas = [
                val_losses,
                test_losses,
            
                val_aps,
                test_aps,
            
                val_rocs,
                test_rocs]
    titles = [
            "Validation_Loss",
            "Test_Loss",
            
            "Validation_Average_Precision",
            "Test_Average_Precision",
            
            "Validation_ROC",
            "Test_ROC"]
    
    for i, ax in enumerate(axes.flat):
        
        data = datas[i]
        title = titles[i]
        epochs = [index for index in range(len(data))] # if i % 3 != 2 else test_epochs
        
        # Plot the data in the current subplot
        ax.plot(epochs, data)
        ax.set_title(title)

    # Adjust spacing between subplots
    fig.tight_layout()
    print(f"Test_Loss: {test_losses[-1]}")
    print(f"Test_ROC: {test_rocs[-1]}")
    print(f"Test_Average_Precision: {test_aps[-1]}")
    print(f"Test_Accuracy: {test_accs[-1]}")
    # TODO: Remove to avoid excessive printing
    return test_losses[-1], test_rocs[-1], test_aps[-1]

def to_poincare(x, c):
    K = 1. / c
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
#     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
    return sqrtK * x.narrow(-1, 1, d) / (x[0] + sqrtK)

def scale_two_dimensional_embeddings_to_poincare_disk(poincare_embeddings):
    eps = 1e-2
    radii = [torch.sqrt(poincare_embedding[0] ** 2 + poincare_embedding[1] ** 2) 
            for poincare_embedding in poincare_embeddings]
    max_radius = np.max(radii)
    for poincare_embedding in poincare_embeddings:
        poincare_embedding[0] /= (max_radius + eps)
        poincare_embedding[1] /= (max_radius + eps)
    return poincare_embeddings

def get_adjacency_matrix_for_binary_tree(depth):
        adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])
        
        for node_index in range(1, (2 ** depth - 1) // 2 + 1):
            adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
            adj_matrix[node_index - 1][2 * node_index] = 1.0
            adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
            adj_matrix[2 * node_index][node_index - 1] = 1.0
            
        return adj_matrix
    
def get_adjacency_matrix_for_binary_cyclic_tree(depth):
    adj_matrix = np.array([[0.0 for _ in range(2 ** depth - 1)] for _ in range(2 ** depth - 1)])

    for node_index in range(1, (2 ** depth - 1) // 2 + 1):
        adj_matrix[node_index - 1][2 * node_index - 1] = 1.0
        adj_matrix[node_index - 1][2 * node_index] = 1.0
        adj_matrix[2 * node_index - 1][node_index - 1] = 1.0
        adj_matrix[2 * node_index][node_index - 1] = 1.0
    for node_index in range(len(adj_matrix)):
        if (node_index + 1) % 2 == 0: 
            adj_matrix[node_index][node_index + 1] = 1.0
            adj_matrix[node_index + 1][node_index] = 1.0
    return adj_matrix
    
def viz_bin_tree(depth, is_cyclic=False):
    if not is_cyclic: adj_matrix = get_adjacency_matrix_for_binary_tree(depth)
    else: adj_matrix = get_adjacency_matrix_for_binary_cyclic_tree(depth)
    G = nx.from_numpy_matrix(adj_matrix)
    nx.draw(G, with_labels=True)
    plt.savefig(f"binary_tree_depth_{depth}.png" if not is_cyclic else f"binary_cyclic_tree_depth_{depth}.png")
    plt.show()

def viz_embeddings(embeddings_path, 
                reduce_dim = False, 
                is_bin_tree= False, 
                is_cyclic= False, 
                use_colors = False,
                use_indices = False,
                show_edges = False,
                use_save = True,
                prefix = ""):
    from utils.data_utils import TREE_DEPTH
    # if is_bin_tree: viz_bin_tree(depth, is_cyclic=is_cyclic)
    embeddings_name = embeddings_path.split('\\')[-1]
    if 'log' in embeddings_name: return 
    hyperboloid_embeddings = np.load(embeddings_path)
    if not reduce_dim:
        c = 1.
        torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
        poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    else:
        from sklearn.manifold import TSNE
        projected_embeddings = TSNE(n_components=2,
                        init='random', perplexity=3).fit_transform(hyperboloid_embeddings)
        poincare_embeddings = torch.from_numpy(projected_embeddings)

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Create a circle with radius 1 centered at the origin
    circ = plt.Circle((0, 0), 
                radius= 1, 
                edgecolor='black', 
                facecolor='None', 
                linewidth=3, 
                alpha=0.5)
    # Add the circle to the axis
    ax.set_aspect(0.9)
    ax.add_patch(circ)
    if is_bin_tree: ax.set_title(f"{prefix}_Embeddings")
    else: ax.set_title(f"{embeddings_name[:-4]}")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    if use_colors: # Coloring for Binary Tree Nodes Onlys
        color_log_indices = np.array([int(np.log2(index + 1)) for index in range(len(poincare_embeddings))])
        cmap = plt.cm.jet   
        poincare_embeddings_x = [poincare_embedding[0] for poincare_embedding in poincare_embeddings]
        poincare_embeddings_y = [poincare_embedding[1] for poincare_embedding in poincare_embeddings]
        plt.scatter(poincare_embeddings_x, poincare_embeddings_y, c=color_log_indices, cmap=cmap)
        if use_indices:
            for index, poincare_embedding in enumerate(poincare_embeddings):
                ax.annotate(index, (poincare_embedding[0], poincare_embedding[1]))

    else:
        for index, poincare_embedding in enumerate(poincare_embeddings):
            plt.scatter(poincare_embedding[0], poincare_embedding[1])

    cbar = plt.colorbar()
    cbar.set_label('Node Depth')
    
    adj_matrix = get_adjacency_matrix_for_binary_tree(TREE_DEPTH)

    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(len(poincare_embeddings))], 
                                        'y': [poincare_embeddings[i][1] for i in range(len(poincare_embeddings))],
                                        'id': [i for i in range(len(poincare_embeddings))]}
                                    )
    if show_edges:
        edge_list_0 = get_edges(adj_matrix)
        for i in range(len(edge_list_0)):
            x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
            x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
            _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.5)
    if use_save: plt.savefig(f"{prefix}_embeddings.png")
    plt.show()




def get_edges(adj_matrix):
        edges = []
        for i in range(len(adj_matrix)):
            for j in range(i, len(adj_matrix)):
                if adj_matrix[i, j] > 0:
                    edges.append([i, j])
        return edges


def plot_embeddings(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    labels = list(set(cortices))
    embeddings_df = get_embeddings_df(embeddings) 
    # Create the legend
    if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
    
    for i in embeddings_df.label.unique():
        emb_L = embeddings_df.loc[(embeddings_df.LR == "L")]
        plt.scatter(emb_L.loc[(emb_L.label == i), 'x'], 
                    emb_L.loc[(emb_L.label == i), 'y'], 
                    c = COLORS[i],
                    s = 50, 
                    marker = "v",)
                    # label = cortex_ids_to_cortices[i]) avoid repeating same labels but with differnet shape
        emb_R = embeddings_df.loc[(embeddings_df.LR == "R")]
        plt.scatter(emb_R.loc[(emb_R.label == i), 'x'], 
                    emb_R.loc[(emb_R.label == i), 'y'], 
                    c = COLORS[i], 
                    s = 50,
                    marker = "s",
                    label = cortex_ids_to_cortices[i])
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    circ = plt.Circle((0, 0), 
                    radius=1, 
                    edgecolor='black', 
                    facecolor='None', 
                    linewidth=3, 
                    alpha=0.5)
    ax.add_patch(circ)
    # plot_edges = False
    # TODO: Need to include cam_can adjacency matrix as input to make plotting edges possible  
    # if plot_edges:
    #     edge_list_0 = get_edges(adj)
    #     for i in range(len(edge_list_0)):
    #         x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
    #         x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
    #         _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
    if title != None:
        plt.title(title, size=16)
    plt.savefig("fhnn_embedding_for_average_592_plv.png")
    
    permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    if use_centroids:
        embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
        embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
        embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
        embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)
        # HK MEANS CENTROIDS
        # clustering_left = plot_centroids(embeddings_df_left_detensored, is_left=True)
        # clustering_right = plot_centroids(embeddings_df_right_detensored, is_left=False)

        # embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
        # plot_hyperbolic_radii(embeddings_df_detensored, 
        #                     "Hyperbolic_Cluster_Radius_CamCan_Avg_276", 
        #                     clustering, 
        #                     permuted_colors=permuted_colors)
        # plot_hyperbolic_cohesion(embeddings_df_detensored, 
        #                         "Hyperbolic_Cluster_Cohesion_CamCan_Avg_276", 
        #                         clustering,
        #                         permuted_colors=permuted_colors)
        # HK MEANS CENTROID RADII 
        # plot_hyperbolic_radii_left_right(embeddings_df_left_detensored,
        #                                 embeddings_df_right_detensored,  
        #                                 clustering_left=clustering_left,
        #                                 clustering_right=clustering_right,
        #                                 use_diff_plot=True, 
        #                                 permuted_colors=permuted_colors)
        plot_avg_hyperbolic_radii_left_right(embeddings_df_left_detensored,
                                        embeddings_df_right_detensored,  
                                        use_diff_plot=True, 
                                        permuted_colors=permuted_colors)
def plot_embeddings_meg(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    labels = list(set(cortices))
    embeddings_df = get_meg_embeddings_df(embeddings) 
    # Create the legend
    if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
    
    cmap = plt.cm.jet

    for i in embeddings_df.label.unique():
        emb_L = embeddings_df.loc[(embeddings_df.LR == "L")]
        plt.scatter(emb_L.loc[(emb_L.label == i), 'x'], 
                    emb_L.loc[(emb_L.label == i), 'y'], 
                    color = cmap(i / len(embeddings_df.label.unique())),
                    s = 50, 
                    marker = "v",)
                    # label = cortex_ids_to_cortices[i]) avoid repeating same labels but with differnet shape
        emb_R = embeddings_df.loc[(embeddings_df.LR == "R")]
        plt.scatter(emb_R.loc[(emb_R.label == i), 'x'], 
                    emb_R.loc[(emb_R.label == i), 'y'], 
                    color = cmap(i / len(embeddings_df.label.unique())), 
                    s = 50,
                    marker = "s",
                    label = embeddings_df.label[i])
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    circ = plt.Circle((0, 0), 
                    radius=1, 
                    edgecolor='black', 
                    facecolor='None', 
                    linewidth=3, 
                    alpha=0.5)
    ax.add_patch(circ)
    # plot_edges = False
    # TODO: Need to include cam_can adjacency matrix as input to make plotting edges possible  
    # if plot_edges:
    #     edge_list_0 = get_edges(adj)
    #     for i in range(len(edge_list_0)):
    #         x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
    #         x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
    #         _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
    if title != None:
        plt.title(title, size=16)
    plt.savefig("meg_embedding_180.png")
    
    # permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    if use_centroids:
        embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
        embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
        embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
        embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)

def plot_total_avg_hyperbolic_radii(embeddings_dir):
    
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    
    total_cortices_to_avg_hyperbolic_cluster_radii_left, total_cortices_to_avg_hyperbolic_cluster_radii_right, permuted_colors = \
        get_total_cortices_to_avg_hyperbolic_cluster_radii_left_right(embeddings_dir, cortices, cortex_ids_to_cortices) 
    
    plot_total_avg_hyperbolic_radii_left_right(total_cortices_to_avg_hyperbolic_cluster_radii_left, total_cortices_to_avg_hyperbolic_cluster_radii_right, permuted_colors)
    

def get_total_cortices_to_avg_hyperbolic_cluster_radii_left_right(embeddings_dir, cortices, cortex_ids_to_cortices, use_scale=False):
    total_cortices_to_avg_hyperbolic_cluster_radii_left = None
    total_cortices_to_avg_hyperbolic_cluster_radii_right = None
    permuted_colors = None
    
    num_test_sbjs = 0
    # TODO: Expand to val and train sets
    for test_embeddings_dir in os.listdir(embeddings_dir):
        if 'test' not in test_embeddings_dir: continue
        _, _, test_index_str = test_embeddings_dir.split("_")
        embeddings = np.load(os.path.join(embeddings_dir, test_embeddings_dir))
        
        labels = list(set(cortices))
        embeddings_df = get_embeddings_df(embeddings) 
        if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
        # Create the legend
        if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
        
        embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
        embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
        embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
        embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)
        
        if not total_cortices_to_avg_hyperbolic_cluster_radii_left:
            total_cortices_to_avg_hyperbolic_cluster_radii_left = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_left_detensored, cortex_ids_to_cortices)
        else:
            cortices_to_avg_hyperbolic_cluster_radii_left = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_left_detensored, cortex_ids_to_cortices)
            for cortex in cortices_to_avg_hyperbolic_cluster_radii_left:
                total_cortices_to_avg_hyperbolic_cluster_radii_left[cortex] += cortices_to_avg_hyperbolic_cluster_radii_left[cortex]
        if not total_cortices_to_avg_hyperbolic_cluster_radii_right:
            total_cortices_to_avg_hyperbolic_cluster_radii_right = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_right_detensored, cortex_ids_to_cortices)
        else:
            cortices_to_avg_hyperbolic_cluster_radii_right = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_right_detensored, cortex_ids_to_cortices)
            for cortex in cortices_to_avg_hyperbolic_cluster_radii_right:
                total_cortices_to_avg_hyperbolic_cluster_radii_right[cortex] += cortices_to_avg_hyperbolic_cluster_radii_right[cortex]
        num_test_sbjs += 1
    
    for cortex in total_cortices_to_avg_hyperbolic_cluster_radii_right:
        total_cortices_to_avg_hyperbolic_cluster_radii_right[cortex] /= num_test_sbjs
    for cortex in total_cortices_to_avg_hyperbolic_cluster_radii_left:
        total_cortices_to_avg_hyperbolic_cluster_radii_left[cortex] /= num_test_sbjs
    return total_cortices_to_avg_hyperbolic_cluster_radii_left, total_cortices_to_avg_hyperbolic_cluster_radii_right, permuted_colors

def plot_tree_embeddings(embeddings, title=None, use_scale=False, plot_radii=False, show_edges=False):
    from utils.data_utils import TREE_DEPTH
    plt.figure()
    # Create a list of labels for the legend
    embeddings_df = get_tree_embeddings_df(embeddings) 
    # Create the legend
    if use_scale: embeddings_df = scale_embeddings_df_to_poincare_disk(embeddings_df)
    
    color_log_indices = np.array([int(np.log2(index + 1)) for index in range(len(embeddings_df))])
    cmap = plt.cm.jet   
    
    plt.scatter(embeddings_df['x'], embeddings_df['y'], c=color_log_indices, cmap=cmap)
    adj_matrix = get_adjacency_matrix_for_binary_cyclic_tree(TREE_DEPTH)
    if show_edges:
        edge_list_0 = get_edges(adj_matrix)
        for i in range(len(edge_list_0)):
            x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
            x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
            _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.5)

    cbar = plt.colorbar()
    cbar.set_label('Node Depth')
    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_aspect(0.9)
    
    circ = plt.Circle((0, 0), 
                    radius=1, 
                    edgecolor='black', 
                    facecolor='None', 
                    linewidth=3, 
                    alpha=0.5)
    ax.add_patch(circ)
    if title != None:
        plt.title(title, size=16)
    plt.savefig("tree_embeddings.png")
    
    # permuted_colors = [color_log_indices[i] for i in embeddings_df.label.unique()]
    if plot_radii:
        embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
        cmap = plt.cm.jet
        color_numbers = [i for i in embeddings_df.label.unique()]
        # Normalize the array of numbers to the range [0, 1]
        normalized_numbers = (color_numbers - np.min(color_numbers)) / (np.max(color_numbers) - np.min(color_numbers))

        # Apply the colormap to the normalized array
        permuted_colors = cmap(normalized_numbers)
        # clustering = plot_tree_centroids(embeddings_df_detensored, is_left=True, permuted_colors=permuted_colors)
        
        # plot_hyperbolic_radii_tree(embeddings_df_detensored,
        #                     clustering=clustering, 
        #                     permuted_colors=permuted_colors)
        plot_avg_hyperbolic_radii_tree(embeddings_df_detensored,
                            # clustering=clustering, 
                            permuted_colors=permuted_colors)

def detensorify_embeddings_df(embeddings_df):
    embeddings_df_detensored = embeddings_df.copy()
    embeddings_df_detensored['x'] = [x.item() for x in embeddings_df_detensored['x']]
    embeddings_df_detensored['y'] = [y.item() for y in embeddings_df_detensored['y']]
    return embeddings_df_detensored

def plot_tree_centroids(embeddings_df_detensored, clustering=None, permuted_colors=None, is_left=False) -> dict:
    # permuted_colors = np.array([int(np.log2(index + 1)) for index in range(len(embeddings_df_detensored))])
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df_detensored.label.unique()]
    if not clustering: clustering = fit_clusters(embeddings_df_detensored, is_tree=True)
    hkmeans = clustering['model']
    if is_left: left_right_marker = "v"
    else: left_right_marker = "s"
    MARKER_SIZE = 100
    for index, centroid in enumerate(hkmeans.centroids):
        # if index == len(hkmeans.centroids) - 1: break
        plt.scatter(centroid[0], 
                    centroid[1], 
                color = permuted_colors[index],
                s = MARKER_SIZE, 
                marker = left_right_marker,
                edgecolors='black')
    return clustering


def plot_centroids(embeddings_df_detensored, clustering=None, permuted_colors=None, is_left=False) -> dict:
    permuted_colors = [COLORS[i] for i in embeddings_df_detensored.label.unique()]
    if not clustering: clustering = fit_clusters(embeddings_df_detensored)
    hkmeans = clustering['model']
    if is_left: left_right_marker = "v"
    else: left_right_marker = "s"
    MARKER_SIZE = 100
    for index, centroid in enumerate(hkmeans.centroids):
        plt.scatter(centroid[0], 
                    centroid[1], 
                c = permuted_colors[index],
                s = MARKER_SIZE, 
                marker = left_right_marker,
                edgecolors='black')
    return clustering

def plot_cluster_metrics(embeddings_df, title=None, clustering=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    plot_between_cluster_metrics(embeddings_df, cortices, clustering)

def plot_hyperbolic_radii(embeddings_df, title=None, clustering=None, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    cortices_to_hyperbolic_radii = get_cortices_to_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices, clustering)
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii.keys())], 
            cortices_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Cortices')
    plt.ylabel('Hyperbolic Cluster Radius')
    plt.title('Hyperbolic Radius of Cortices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius.png")
    plt.show()

def plot_hyperbolic_radii_tree(embeddings_df, title=None, clustering=None, permuted_colors=None):
    depths_to_hyperbolic_radii = get_depths_to_hyperbolic_cluster_radii(embeddings_df, clustering, is_tree=True)
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar(depths_to_hyperbolic_radii.keys(), 
            depths_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Node Tree Depth')
    plt.ylabel('Hyperbolic Cluster Radius')
    plt.title('Hyperbolic Radius of Depth Node Clusters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius.png")
    plt.show()

def plot_avg_hyperbolic_radii_tree(embeddings_df, title=None, clustering=None, permuted_colors=None):
    
    depths_to_hyperbolic_radii = get_depths_to_avg_hyperbolic_cluster_radii(embeddings_df)
    if permuted_colors is None: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar(depths_to_hyperbolic_radii.keys(), 
            depths_to_hyperbolic_radii.values(), color=permuted_colors)
    plt.xlabel('Node Tree Depth')
    plt.ylabel('Hyperbolic Average Radius')
    plt.title('Hyperbolic Average Radius of Depth Node Clusters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_average_radius.png")
    plt.show()

def plot_total_avg_hyperbolic_radii_left_right(cortices_to_hyperbolic_radii_left, cortices_to_hyperbolic_radii_right, permuted_colors):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    NUM_SUBNETS = 22
    num_subnets = len(cortices_to_hyperbolic_radii_right.keys())
    axes[0].set_xticks(np.arange(num_subnets), rotation=45)
    axes[0].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[0].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            cortices_to_hyperbolic_radii_left.values(), color=permuted_colors)
    axes[0].set_xlabel('Cortices')
    axes[0].set_ylabel('Hyperbolic Average Radius')
    axes[0].set_title('Test Hyperbolic Average Radius of Cortices LEFT')
    axes[0].set_ylim([0, 4])

    axes[1].set_xticks(np.arange(num_subnets))
    axes[1].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[1].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], 
            cortices_to_hyperbolic_radii_right.values(), color=permuted_colors)    
    axes[1].set_xlabel('Cortices')
    axes[1].set_ylabel('Hyperbolic Average Radius')
    axes[1].set_title('Test Hyperbolic Average Radius of Cortices RIGHT')
    axes[1].set_ylim([0, 4])

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig("hyperbolic_average_radius_left_right.png")
    plt.show()

    # Difference Plot for ROIs
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            np.array([*cortices_to_hyperbolic_radii_left.values()]) - np.array([*cortices_to_hyperbolic_radii_right.values()]),
            color = permuted_colors, 
            label  = 'Left - Right Hemisphere')
    ax.set_xlabel('Cortex')
    ax.set_ylabel('Hyperbolic Average Radius Difference')
    ax.set_title("Hyperbolic Average Difference of Left Minus Right Hemisphere Cortices")
    # ax.set_ylim(-1, 1)
    plt.xticks(rotation=45)
    plt.savefig("hyperbolic_average_radius_left_right_diff.png")
    plt.show()


def plot_avg_hyperbolic_radii_left_right(embeddings_df_left, embeddings_df_right, use_diff_plot=False, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()    
    cortices_to_hyperbolic_radii_left = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_right = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices)
    
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df_left.label.unique()]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    NUM_SUBNETS = 22
    num_subnets = len(cortices_to_hyperbolic_radii_right.keys())
    axes[0].set_xticks(np.arange(num_subnets), rotation=45)
    axes[0].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[0].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            cortices_to_hyperbolic_radii_left.values(), color=permuted_colors)
    axes[0].set_xlabel('Cortices')
    axes[0].set_ylabel('Hyperbolic Average Radius')
    axes[0].set_title('Hyperbolic Average Radius of Cortices LEFT')
    # axes[0].set_ylim([0, 3])

    axes[1].set_xticks(np.arange(num_subnets))
    axes[1].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[1].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], 
            cortices_to_hyperbolic_radii_right.values(), color=permuted_colors)    
    axes[1].set_xlabel('Cortices')
    axes[1].set_ylabel('Hyperbolic Average Radius')
    axes[1].set_title('Hyperbolic Average Radius of Cortices RIGHT')
    # axes[1].set_ylim([0, 3])

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig("hyperbolic_average_radius_left_right.png")
    plt.show()

    if use_diff_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
                np.array([*cortices_to_hyperbolic_radii_left.values()]) - np.array([*cortices_to_hyperbolic_radii_right.values()]),
                color = permuted_colors, 
                label  = 'Left - Right Hemisphere')
        ax.set_xlabel('Cortex')
        ax.set_ylabel('Hyperbolic Average Radius Difference')
        ax.set_title("Hyperbolic Average Difference of Left Minus Right Hemisphere Cortices")
        # ax.set_ylim(-1, 1)
        plt.xticks(rotation=45)
        plt.savefig("hyperbolic_average_radius_left_right_diff.png")
        plt.show()


def plot_hyperbolic_radii_left_right(embeddings_df_left, embeddings_df_right, clustering_left=None, clustering_right=None, use_diff_plot=False, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    # cortices_to_hyperbolic_radii_left, cortices_to_hyperbolic_radii_right = \
        # get_cortices_to_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices)
    cortices_to_hyperbolic_radii_left = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_left, cortex_ids_to_cortices, clustering_left)
    cortices_to_hyperbolic_radii_right = get_cortices_to_hyperbolic_cluster_radii(embeddings_df_right, cortex_ids_to_cortices, clustering_right)
    
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df_left.label.unique()]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
    NUM_SUBNETS = 22
    num_subnets = len(cortices_to_hyperbolic_radii_right.keys())
    axes[0].set_xticks(np.arange(num_subnets), rotation=45)
    axes[0].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[0].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
            cortices_to_hyperbolic_radii_left.values(), color=permuted_colors)
    axes[0].set_xlabel('Cortices')
    axes[0].set_ylabel('Hyperbolic Cluster Radius')
    axes[0].set_title('Hyperbolic Radius of Cortices LEFT')
    # axes[0].set_ylim([0, 3])

    axes[1].set_xticks(np.arange(num_subnets))
    axes[1].set_xticklabels([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], rotation=45)
    axes[1].bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_right.keys())], 
            cortices_to_hyperbolic_radii_right.values(), color=permuted_colors)    
    axes[1].set_xlabel('Cortices')
    axes[1].set_ylabel('Hyperbolic Cluster Radius')
    axes[1].set_title('Hyperbolic Radius of Cortices RIGHT')
    # axes[1].set_ylim([0, 3])

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_radius_left_right.png")
    plt.show()

    if use_diff_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_radii_left.keys())], 
                np.array([*cortices_to_hyperbolic_radii_left.values()]) - np.array([*cortices_to_hyperbolic_radii_right.values()]),
                color = permuted_colors, 
                label  = 'Left - Right Hemisphere')
        ax.set_xlabel('Cortex')
        ax.set_ylabel('Hyperbolic Cluster Radius Difference')
        ax.set_title("Hyperbolic Radius Difference of Left Minus Right Hemisphere Cortices")
        # ax.set_ylim(-1, 1)
        plt.xticks(rotation=45)
        plt.savefig("hyperbolic_cluster_radius_left_right_diff.png")
        plt.show()

def plot_hyperbolic_cohesion(embeddings_df, title=None, clustering=None, permuted_colors=None):
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    cortices_to_hyperbolic_cohesions = get_cortices_to_hyperbolic_cluster_cohesion(embeddings_df, cortex_ids_to_cortices, clustering)
    if not permuted_colors: permuted_colors = [COLORS[i] for i in embeddings_df.label.unique()]
    # Plotting Radii
    plt.figure(figsize=(10, 6))
    plt.bar([*map(lambda x : CORTEX_TO_ABBREVIATION[x], cortices_to_hyperbolic_cohesions.keys())], 
            cortices_to_hyperbolic_cohesions.values(), color=permuted_colors)
    plt.xlabel('Cortices')
    plt.ylabel('Hyperbolic Cluster Cohesion')
    plt.title('Hyperbolic Cohesion of Cortices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("hyperbolic_cluster_cohesion.png")
    plt.show()

def plot_embeddings_by_parts(embeddings, title=None, use_scale=False, use_cluster_metrics=False, use_centroids=False):
    # Create a list of labels for the legend
    cortices, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    labels = list(set(cortices))
    embeddings_df = get_embeddings_df(embeddings) 
    # Create the legend
    partial_labels = np.array_split(embeddings_df.label.unique(), 5)
    for split_labels in partial_labels:
        embeddings_partial_df = pd.concat([embeddings_df.loc[(embeddings_df.label == i)] for i in split_labels])

        if use_scale: embeddings_partial_df = scale_embeddings_df_to_poincare_disk(embeddings_partial_df)
        
        for i in embeddings_partial_df.label.unique():
            emb_L = embeddings_partial_df.loc[(embeddings_partial_df.LR == "L")]
            plt.scatter(emb_L.loc[(emb_L.label == i), 'x'], 
                        emb_L.loc[(emb_L.label == i), 'y'], 
                        c = COLORS[i],
                        s = 50, 
                        marker = "v",)
            emb_R = embeddings_partial_df.loc[(embeddings_partial_df.LR == "R")]
            plt.scatter(emb_R.loc[(emb_R.label == i), 'x'], 
                        emb_R.loc[(emb_R.label == i), 'y'], 
                        c = COLORS[i], 
                        s = 50,
                        marker = "s",
                        label = cortex_ids_to_cortices[i])
        # Shrink current axis by 20%
        ax = plt.gca()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        circ = plt.Circle((0, 0), 
                        radius=1, 
                        edgecolor='black', 
                        facecolor='None', 
                        linewidth=3, 
                        alpha=0.5)
        ax.add_patch(circ)
        # plot_edges = False
        # TODO: Need to include cam_can adjacency matrix as input to make plotting edges possible  
        # if plot_edges:
        #     edge_list_0 = get_edges(adj)
        #     for i in range(len(edge_list_0)):
        #         x1 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][0]), ['x', 'y']].values[0]
        #         x2 = embeddings_df.loc[(embeddings_df.id == edge_list_0[i][1]), ['x', 'y']].values[0]
        #         _ = plt.plot([x1[0], x2[0]], [x1[1], x2[1]], '--', c='black', linewidth=1, alpha=0.25)
        if title != None:
            plt.title(title, size=16)
        plt.savefig("fhnn_embedding_for_average_276_plv.png")
        
        permuted_colors = [COLORS[i] for i in embeddings_partial_df.label.unique()]
        if use_centroids:
            embeddings_df_left = embeddings_df[embeddings_df.LR == "L"]
            embeddings_df_right = embeddings_df[embeddings_df.LR == "R"]
            embeddings_df_left_detensored = detensorify_embeddings_df(embeddings_df_left)
            embeddings_df_right_detensored = detensorify_embeddings_df(embeddings_df_right)
            plot_centroids(embeddings_df_left_detensored)
            plot_centroids(embeddings_df_right_detensored)

            embeddings_df_detensored = detensorify_embeddings_df(embeddings_df)
            
            plot_hyperbolic_radii_left_right(embeddings_df_detensored,
                                            "Hyperbolic_Cluster_Radius_LR_CamCan_Avg_276",
                                            use_diff_plot=True,
                                            permuted_colors=permuted_colors)

def viz_hyperbolic_radii_box_plots(date):
    embeddings_df_list = get_embeddings_df_list(date)
    cortices_to_avg_hyp_radii_distribution = dict()
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    for embeddings_df in embeddings_df_list:
        cortices_to_avg_hyp_radii = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices)
        for cortex in cortices_to_avg_hyp_radii:
            cortices_to_avg_hyp_radii_distribution[cortex] = cortices_to_avg_hyp_radii_distribution.get(cortex, []) + [cortices_to_avg_hyp_radii[cortex]]
    
    rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
    
    bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution.values(), 
                patch_artist=True, 
                notch=False)
    for box, color in zip(bp['boxes'], rgb_colors):
        box.set(facecolor=color)

    plt.title("Box Plot of Hyperbolic Cluster Radii for Cortex Subregions")
    plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution) + 1), 
                [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution],
                rotation=90)
    plt.show()


def viz_hyperbolic_radii_box_plots_by_decade(date):
    for nth_index in range(NUM_DECADES):
        embeddings_df_list_decade = get_embeddings_df_list_by_decade(date, nth_index)
        cortices_to_avg_hyp_radii_distribution = dict()
        _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
        decade_position_str = DECADE_POSITIONS_STR_LIST[nth_index]

        for embeddings_df in embeddings_df_list_decade:
            cortices_to_avg_hyp_radii = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices)
            for cortex in cortices_to_avg_hyp_radii:
                cortices_to_avg_hyp_radii_distribution[cortex] = cortices_to_avg_hyp_radii_distribution.get(cortex, []) + [cortices_to_avg_hyp_radii[cortex]]
        
        rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
        
        plt.figure()
        plt.ylim(0, 4.5)
        bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution.values(), 
                    patch_artist=True, 
                    notch=True)
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)

        plt.title(f"Box Plot of {decade_position_str} Decade Hyperbolic Cluster Radii for Cortex Subregions")
        plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution) + 1), 
                    [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution],
                    rotation=90)
        plt.show()

def viz_hyperbolic_radii_box_plots_by_decade_left_right(date):
    for nth_index in range(NUM_DECADES):
        embeddings_df_list_decade = get_embeddings_df_list_by_decade(date, nth_index)
        cortices_to_avg_hyp_radii_distribution_L = dict()
        cortices_to_avg_hyp_radii_distribution_R = dict()
        _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
        decade_position_str = DECADE_POSITIONS_STR_LIST[nth_index]

        for embeddings_df in embeddings_df_list_decade:
            cortices_to_avg_hyp_radii_L, cortices_to_avg_hyp_radii_R = get_cortices_to_avg_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices)
            for cortex in cortices_to_avg_hyp_radii_L:
                cortices_to_avg_hyp_radii_distribution_L[cortex] = cortices_to_avg_hyp_radii_distribution_L.get(cortex, []) + [cortices_to_avg_hyp_radii_L[cortex]]
            for cortex in cortices_to_avg_hyp_radii_R:
                cortices_to_avg_hyp_radii_distribution_R[cortex] = cortices_to_avg_hyp_radii_distribution_R.get(cortex, []) + [cortices_to_avg_hyp_radii_R[cortex]]
        
        rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
        
        bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution_L.values(), 
                    patch_artist=True, 
                    notch=True)
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)

        plt.title(f"{decade_position_str} Decade Left Hemisphere Hyperbolic Radii for Cortex Subnetworks")
        plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution_L) + 1), 
                    [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution_L],
                    rotation=90)
        plt.ylim(0, 4.5)
        plt.show()

        plt.figure()
        bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution_R.values(), 
                    patch_artist=True, 
                    notch=True)
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)

        plt.title(f"{decade_position_str} Decade Right Hemisphere Hyperbolic Radii for Cortex Subnetworks")
        plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution_R) + 1), 
                    [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution_R],
                    rotation=90)
        plt.ylim(0, 4.5)
        plt.show()


def viz_hyperbolic_radii_box_plots_left_right(date):
    embeddings_df_list = get_embeddings_df_list(date) # Gets only test embeddings from folder directory
    cortices_to_avg_hyp_radii_distribution_L = dict()
    cortices_to_avg_hyp_radii_distribution_R = dict()
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()

    for embeddings_df in embeddings_df_list:
        cortices_to_avg_hyp_radii_L, cortices_to_avg_hyp_radii_R = get_cortices_to_avg_hyperbolic_cluster_radii_left_right(embeddings_df, cortex_ids_to_cortices)
        for cortex in cortices_to_avg_hyp_radii_L:
            cortices_to_avg_hyp_radii_distribution_L[cortex] = cortices_to_avg_hyp_radii_distribution_L.get(cortex, []) + [cortices_to_avg_hyp_radii_L[cortex]]
        for cortex in cortices_to_avg_hyp_radii_R:
            cortices_to_avg_hyp_radii_distribution_R[cortex] = cortices_to_avg_hyp_radii_distribution_R.get(cortex, []) + [cortices_to_avg_hyp_radii_R[cortex]]
    
    rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
    
    bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution_L.values(), 
                patch_artist=True, 
                notch=True)
    for box, color in zip(bp['boxes'], rgb_colors):
        box.set(facecolor=color)

    plt.title("Box Plot of LEFT Hemisphere Hyperbolic Cluster Radii for Cortex Subregions")
    plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution_L) + 1), 
                [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution_L],
                rotation=90)
    plt.ylim(0, 4.5)
    plt.show()

    plt.figure()
    bp = plt.boxplot(cortices_to_avg_hyp_radii_distribution_R.values(), 
                patch_artist=True, 
                notch=True)
    for box, color in zip(bp['boxes'], rgb_colors):
        box.set(facecolor=color)

    plt.title("Box Plot of RIGHT Hemisphere Hyperbolic Cluster Radii for Cortex Subregions")
    plt.xticks(range(1, len(cortices_to_avg_hyp_radii_distribution_R) + 1), 
                [CORTEX_TO_ABBREVIATION[cortex] for cortex in cortices_to_avg_hyp_radii_distribution_R],
                rotation=90)
    plt.ylim(0, 4.5)
    plt.show()

def viz_subnetwork_radii_across_age(date, precalculated_radii=None):
    from hyperbolic_clustering.hyperbolic_cluster_metrics import get_subnetwork_hyperbolic_radii_per_sbj_left_right
    subnetwork_hyperbolic_radii_per_sbj_L, subnetwork_hyperbolic_radii_per_sbj_R = get_subnetwork_hyperbolic_radii_per_sbj_left_right(date, precalculated_radii=precalculated_radii)
    
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    subnets_to_decades_to_hyperbolic_radii_L = [dict() for _ in range(NUM_SUBNETS)]
    subnets_to_decades_to_hyperbolic_radii_R = [dict() for _ in range(NUM_SUBNETS)]
    # rgb_colors = [mcolors.hex2color(COLORS[subnet_num]) for subnet_num in range(1, NUM_SUBNETS + 1)] # COLORS are 1 indexed due to HCP-MMP1 being 1-indexed
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]

    decade_sbj_num_lists = [[sbj_num for sbj_num in range(NUM_SBJS) if DECADES[nth_index] <= age_labels[sbj_num] < DECADES[nth_index + 1]] for nth_index in range(NUM_DECADES - 1)]
    for subnet_num in range(NUM_SUBNETS):
        
        for nth_index, decade_sbj_num_list in enumerate(decade_sbj_num_lists):
            subnet_radii_L = [subnetwork_hyperbolic_radii_per_sbj_L[sbj_num][subnet_num] for sbj_num in decade_sbj_num_list]
            subnet_radii_R = [subnetwork_hyperbolic_radii_per_sbj_R[sbj_num][subnet_num] for sbj_num in decade_sbj_num_list]
        
            subnets_to_decades_to_hyperbolic_radii_L[subnet_num][nth_index] = subnet_radii_L
            subnets_to_decades_to_hyperbolic_radii_R[subnet_num][nth_index] = subnet_radii_R
                
    
    for subnet_num in range(NUM_SUBNETS):
        rgb_color = rgb_colors[subnet_num]
        cortex = cortex_ids_to_cortices[subnet_num + 1]
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Left {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([subnets_to_decades_to_hyperbolic_radii_L[subnet_num][nth_index] for nth_index in range(NUM_DECADES - 1)], 
                    patch_artist=True, 
                    notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)
        
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Right {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([subnets_to_decades_to_hyperbolic_radii_R[subnet_num][nth_index] for nth_index in range(NUM_DECADES - 1)], 
                        patch_artist=True, 
                        notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)

def viz_cortex_hyperbolic_radii_across_age(date):
    embeddings_df_list_with_age_labels = get_embeddings_df_list_with_age_labels(date)
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    
    cortices_to_hyperbolic_radii_across_age_L, cortices_to_hyperbolic_radii_across_age_R = \
        get_cortex_regions_to_hyperbolic_radii_across_age_left_right(embeddings_df_list_with_age_labels, cortex_ids_to_cortices)

    # TODO: FIX TO CORRECT AGE LABELS AND PLOT WITH AGE LABELS
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    
    embeddings_age_labels = [embeddings_df.age_label[0] for embeddings_df in embeddings_df_list_with_age_labels]
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    
    rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
    for cortex_id in range(1, NUM_SUBNETS + 1):
        decades_to_hyperbolic_radii_L = dict()
        decades_to_hyperbolic_radii_R = dict()    
        rgb_color = rgb_colors[cortex_id - 1]

        for nth_index in range(NUM_DECADES - 1):
            lower_bound = DECADES[nth_index]
            upper_bound = DECADES[nth_index + 1]
            decade_position_str = DECADE_POSITIONS_STR_LIST[nth_index]
            
            cortex = cortex_ids_to_cortices[cortex_id]
            age_labels_to_hyperbolic_radii_L = dict()
            age_labels_to_hyperbolic_radii_R = dict()
            
            for age_label, hyp_radii_L, hyp_radii_R in zip(
                                                        embeddings_age_labels, 
                                                        cortices_to_hyperbolic_radii_across_age_L[cortex], 
                                                        cortices_to_hyperbolic_radii_across_age_R[cortex]
                                                        ): 
                if not lower_bound <= age_label < upper_bound: continue 
                print("THESE ARE THE AGE LABELS", age_label)
                decades_to_hyperbolic_radii_L[nth_index] = decades_to_hyperbolic_radii_L.get(nth_index, []) + [hyp_radii_L]
                decades_to_hyperbolic_radii_R[nth_index] = decades_to_hyperbolic_radii_R.get(nth_index, []) + [hyp_radii_R]
            
        # TODO: Box Plot per Decade
        
        # sorted_embeddings_age_labels = sorted(embeddings_age_labels)
        # sorted_hyperbolic_radii_L = [age_labels_to_hyperbolic_radii_L[age_label] for age_label in sorted_embeddings_age_labels]
        # sorted_hyperbolic_radii_R = [age_labels_to_hyperbolic_radii_R[age_label] for age_label in sorted_embeddings_age_labels]
        
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Left {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([decades_to_hyperbolic_radii_L[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                    patch_artist=True, 
                    notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)
        
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Right {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([decades_to_hyperbolic_radii_R[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                        patch_artist=True, 
                        notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)

        use_plot_means = False
        if use_plot_means:
            sorted_embeddings_age_labels = sorted(embeddings_age_labels)
            sorted_means_L = [sum(age_labels_to_hyperbolic_radii_L[age_label]) / len(age_labels_to_hyperbolic_radii_L[age_label]) for age_label in sorted_embeddings_age_labels]
            sorted_means_R = [sum(age_labels_to_hyperbolic_radii_R[age_label]) / len(age_labels_to_hyperbolic_radii_R[age_label]) for age_label in sorted_embeddings_age_labels]
            
            plt.figure()
            plt.title(f"Hyperbolic Radius Across Age : Left {cortex}")
            plt.ylim(0, 4.5)
            plt.plot(sorted_embeddings_age_labels, sorted_means_L)
            
            plt.figure()
            plt.title(f"Hyperbolic Radius Across Age : Right {cortex}")
            plt.ylim(0, 4.5)
            plt.plot(sorted_embeddings_age_labels, sorted_means_R)

def get_cortex_hyperbolic_radii_across_age(date):
    embeddings_df_list_with_age_labels = get_embeddings_df_list_with_age_labels(date)
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    
    cortices_to_hyperbolic_radii_across_age_L, cortices_to_hyperbolic_radii_across_age_R = \
        get_cortex_regions_to_hyperbolic_radii_across_age_left_right(embeddings_df_list_with_age_labels, cortex_ids_to_cortices)

    # TODO: FIX TO CORRECT AGE LABELS AND PLOT WITH AGE LABELS
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    
    embeddings_age_labels = [embeddings_df.age_label[0] for embeddings_df in embeddings_df_list_with_age_labels]
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    
    rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]
    for cortex_id in range(1, NUM_SUBNETS + 1):
        decades_to_hyperbolic_radii_L = dict()
        decades_to_hyperbolic_radii_R = dict()    
        rgb_color = rgb_colors[cortex_id - 1]

        for nth_index in range(NUM_DECADES - 1):
            lower_bound = DECADES[nth_index]
            upper_bound = DECADES[nth_index + 1]
            decade_position_str = DECADE_POSITIONS_STR_LIST[nth_index]
            
            cortex = cortex_ids_to_cortices[cortex_id]
            age_labels_to_hyperbolic_radii_L = dict()
            age_labels_to_hyperbolic_radii_R = dict()
            
            for age_label, hyp_radii_L, hyp_radii_R in zip(
                                                        embeddings_age_labels, 
                                                        cortices_to_hyperbolic_radii_across_age_L[cortex], 
                                                        cortices_to_hyperbolic_radii_across_age_R[cortex]
                                                        ): 
                if not lower_bound <= age_label < upper_bound: continue 
                print("THESE ARE THE AGE LABELS", age_label)
                decades_to_hyperbolic_radii_L[nth_index] = decades_to_hyperbolic_radii_L.get(nth_index, []) + [hyp_radii_L]
                decades_to_hyperbolic_radii_R[nth_index] = decades_to_hyperbolic_radii_R.get(nth_index, []) + [hyp_radii_R]
            
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Left {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([decades_to_hyperbolic_radii_L[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                    patch_artist=True, 
                    notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)
        
        plt.figure()
        plt.title(f"Hyperbolic Radius Across Age : Right {cortex}")
        plt.ylim(0, 4.5)
        plt.xlabel("Age Decade")
        plt.ylabel("ROI Hyperbolic Radius")
        bp = plt.boxplot([decades_to_hyperbolic_radii_R[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                        patch_artist=True, 
                        notch=True)
        plt.xticks(range(1, NUM_DECADES), 
                    ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                    rotation=45)

        for box in bp['boxes']:
            box.set(facecolor=rgb_color)

        use_plot_means = False
        if use_plot_means:
            sorted_embeddings_age_labels = sorted(embeddings_age_labels)
            sorted_means_L = [sum(age_labels_to_hyperbolic_radii_L[age_label]) / len(age_labels_to_hyperbolic_radii_L[age_label]) for age_label in sorted_embeddings_age_labels]
            sorted_means_R = [sum(age_labels_to_hyperbolic_radii_R[age_label]) / len(age_labels_to_hyperbolic_radii_R[age_label]) for age_label in sorted_embeddings_age_labels]
            
            plt.figure()
            plt.title(f"Hyperbolic Radius Across Age : Left {cortex}")
            plt.ylim(0, 4.5)
            plt.plot(sorted_embeddings_age_labels, sorted_means_L)
            
            plt.figure()
            plt.title(f"Hyperbolic Radius Across Age : Right {cortex}")
            plt.ylim(0, 4.5)
            plt.plot(sorted_embeddings_age_labels, sorted_means_R)


# for age_label in embeddings_age_labels:
#     plt.scatter(age_label, sum(age_labels_to_hyperbolic_radii_L[age_label]) / len(age_labels_to_hyperbolic_radii_L[age_label]))
#     plt.xlabel("Age")
#     plt.ylabel("ROI Hyperbolic Radius")

# plt.figure()
# plt.title(f"Box Plot of RIGHT Hemisphere Across Age for Cortex Subregion : {cortex}")
# plt.ylim(0, 4.5)
# for age_label in embeddings_age_labels:
#     plt.scatter(age_label, sum(age_labels_to_hyperbolic_radii_R[age_label]) / len(age_labels_to_hyperbolic_radii_R[age_label]))
#     plt.xlabel("Age")
#     plt.ylabel("ROI Hyperbolic Radius")


# Create Box Plots from Every Age Label in age_labels_to_hyperbolic_radii_L
# plt.figure()

# plt.title(f"Box Plot of LEFT Hemisphere Across Age for Cortex Subregion : {cortex}")
# plt.xticks(range(1, len(embeddings_age_labels) + 1), 
#             embeddings_age_labels,
#             rotation=90)
# for age_label in embeddings_age_labels:
#     bp = plt.boxplot(age_labels_to_hyperbolic_radii_L[age_label], 
#             patch_artist=True, 
#             notch=True)
# plt.xlabel("Age")
# plt.ylabel("ROI Hyperbolic Radius")

# plt.figure()
# plt.ylim(0, 4.5)
# plt.title(f"Box Plot of RIGHT Hemisphere Across Age for Cortex Subregion : {cortex}")
# plt.xticks(range(1, len(embeddings_age_labels) + 1), 
#             embeddings_age_labels,
#             rotation=90)
# for age_label in embeddings_age_labels:
#     bp = plt.boxplot(age_labels_to_hyperbolic_radii_R[age_label]    , 
#             patch_artist=True, 
#             notch=True)
# plt.xlabel("Age")
# plt.ylabel("ROI Hyperbolic Radius")

def viz_total_edges_box_plots_by_decade():
    age_labels_to_total_edges = get_age_labels_to_total_edges_by_threshold(FIVE_PERCENT_THRESHOLD)
    cmap = plt.cm.jet        
    print(age_labels_to_total_edges, "LABELS TO TOTAL EDGES!!!!!!!!!!!")
    # TODO: FIX TO CORRECT AGE LABELS AND PLOT WITH AGE LABELS
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    
    # embeddings_age_labels = [embeddings_df.age_label[0] for embeddings_df in embeddings_df_list_with_age_labels]
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    
    rgb_colors = plt.cm.jet(np.linspace(0, 1, len(DECADES)))
    decades_to_total_edges = dict()
    
    for nth_index in range(NUM_DECADES - 1):
        lower_bound = DECADES[nth_index]
        upper_bound = DECADES[nth_index + 1]
        rgb_color = rgb_colors[nth_index]
        
        for age_label in age_labels:
            print("THESE ARE THE AGE LABELS", age_label)
            if not lower_bound <= age_label < upper_bound: continue 
            for total_edges in age_labels_to_total_edges[age_label]:
                decades_to_total_edges[nth_index] = decades_to_total_edges.get(nth_index, []) + [total_edges]
    print(decades_to_total_edges, "DECADES TO TOTAL EDGES!!!!!!!!!!!")
    plt.figure()
    plt.title(f"Total Number of Edges")
    # plt.ylim(0, 4.5)
    plt.xlabel("Age Decade")
    plt.ylabel("Level of Connectivity (Measured by Number of Edges)")
    bp = plt.boxplot([decades_to_total_edges[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                patch_artist=True, 
                notch=True)
    plt.xticks(range(1, NUM_DECADES), 
                ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                rotation=45)

    for box_index, box in enumerate(bp['boxes']):
        box.set(facecolor=rgb_colors[box_index])

def get_average_roi_hyperbolic_radii_per_sbj_across_runs(date):
    """
    Acquire a 587 x 360 matrix (dictionary with 587 keys mapping to lists of 360) of average hyperbolic radii across runs (NUM_SBJS x NUM_ROIS)
    """
    embeddings_df_list = get_embeddings_df_list_all_splits(date)
    print("THIS IS THE LENGTH OF EMBEDDINGS LIST", len(embeddings_df_list))
    average_roi_hyperbolic_radii_list_per_sbj = dict()
    for sbj_num in range(NUM_SBJS):
        roi_hyperbolic_radii_vectors_from_same_sbj_num = [np.array(get_roi_hyperbolic_radii_list(embeddings_df)) \
                                                        for embeddings_df in embeddings_df_list if embeddings_df['sbj_num'][0] == sbj_num]
        average_roi_hyperbolic_radii_list_from_same_sbj_num = sum(roi_hyperbolic_radii_vectors_from_same_sbj_num) / len(roi_hyperbolic_radii_vectors_from_same_sbj_num)
        
        average_roi_hyperbolic_radii_list_per_sbj[sbj_num] = average_roi_hyperbolic_radii_list_from_same_sbj_num
    return average_roi_hyperbolic_radii_list_per_sbj

def get_average_hyperbolic_radii_per_sbj_across_rois(date):    
    """
    587 x 1
    Returns a NUM_SBJS-long list of the averaged-360-ROI hyperbolic radii for each subject 
    Useful for t-testing the difference between two sets of hyperbolic radii
    Useful for spearman correlation between hyperbolic radii and graph-theoretic measures
    """
    hcp_atlas_df = get_hcp_atlas_df()
    cortices = hcp_atlas_df['cortex']
    cortex_ids = hcp_atlas_df['Cortex_ID']
    cortex_ids_to_cortices = {cortex_ids[i] : cortices[i] for i in range(NUM_ROIS)}

    embeddings_df_list = get_embeddings_df_list_all_splits(date)
    hyperbolic_radii = []
    for sbj_num in range(NUM_SBJS):
        count_embeddings_df_with_same_sbj_num = 0
        avg_radius_sbj = 0
        print("CHECKING SBJ NUM", sbj_num)
        for embeddings_df in embeddings_df_list:
            if embeddings_df['sbj_num'][0] == sbj_num:
                count_embeddings_df_with_same_sbj_num += 1
                cortices_to_hyp_radii = get_cortices_to_avg_hyperbolic_cluster_radii(embeddings_df, cortex_ids_to_cortices)
                avg_radius_sbj += sum(cortices_to_hyp_radii.values()) / len(cortices_to_hyp_radii.values())
        print("NUMBER OF REPEATS", count_embeddings_df_with_same_sbj_num)
        avg_radius_sbj /= count_embeddings_df_with_same_sbj_num
        hyperbolic_radii.append(avg_radius_sbj)
    return hyperbolic_radii


def t_test_difference(date_1, date_2):
    from scipy import stats
    
    hyperbolic_radii_1 = get_average_hyperbolic_radii_per_sbj_across_rois(date_1) # 592
    hyperbolic_radii_2 = get_average_hyperbolic_radii_per_sbj_across_rois(date_2) # 592 

    print(hyperbolic_radii_1, "DATA 1")
    print(hyperbolic_radii_2, "DATA 2")

    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(hyperbolic_radii_1, hyperbolic_radii_2)

    # Print the results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # Determine significance level
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: The distributions are statistically significantly different.")
    else:
        print("Fail to reject the null hypothesis: The distributions are not statistically significantly different.")
    return t_statistic, p_value

def viz_myelination_box_plots_by_decade(roi_index):
    from hyperbolic_clustering.hyperbolic_cluster_metrics import get_age_labels_to_myelination
    age_labels_to_myelins = get_age_labels_to_myelination(roi_index)   
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    
    rgb_colors = plt.cm.jet(np.linspace(0, 1, len(DECADES)))
    decades_to_myelins = dict()
    
    for nth_index in range(NUM_DECADES - 1):
        lower_bound = DECADES[nth_index]
        upper_bound = DECADES[nth_index + 1]
        rgb_color = rgb_colors[nth_index]
        
        for age_label in age_labels:
            if not lower_bound <= age_label < upper_bound: continue 
            for total_edges in age_labels_to_myelins[age_label]:
                decades_to_myelins[nth_index] = decades_to_myelins.get(nth_index, []) + [total_edges]
    
    plt.figure()
    plt.title(f"ROI : {ROI_NAMES_360[roi_index]} Myelination Across Age")
    # plt.ylim(0, 4.5)
    plt.xlabel("Age Decade")
    plt.ylabel("Cortical Myelination")
    bp = plt.boxplot([decades_to_myelins[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                patch_artist=True, 
                notch=True)
    plt.xticks(range(1, NUM_DECADES), 
                ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                rotation=45)

    for box_index, box in enumerate(bp['boxes']):
        box.set(facecolor=rgb_colors[box_index])

def viz_thickness_box_plots_by_decade(roi_index):
    from hyperbolic_clustering.hyperbolic_cluster_metrics import get_age_labels_to_thickness
    age_labels_to_thickness = get_age_labels_to_thickness(roi_index)   
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
    
    rgb_colors = plt.cm.jet(np.linspace(0, 1, len(DECADES)))
    decades_to_thickness = dict()
    
    for nth_index in range(NUM_DECADES - 1):
        lower_bound = DECADES[nth_index]
        upper_bound = DECADES[nth_index + 1]
        rgb_color = rgb_colors[nth_index]
        
        for age_label in age_labels:
            if not lower_bound <= age_label < upper_bound: continue 
            for total_edges in age_labels_to_thickness[age_label]:
                decades_to_thickness[nth_index] = decades_to_thickness.get(nth_index, []) + [total_edges]
    
    plt.figure()
    plt.title(f"ROI : {ROI_NAMES_360[roi_index]} Thickness Across Age")
    # plt.ylim(0, 4.5)
    plt.xlabel("Age Decade")
    plt.ylabel("Cortical Thickness")
    bp = plt.boxplot([decades_to_thickness[nth_index] for nth_index in range(NUM_DECADES - 1)], 
                patch_artist=True, 
                notch=True)
    plt.xticks(range(1, NUM_DECADES), 
                ["18-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90"],
                rotation=45)

    for box_index, box in enumerate(bp['boxes']):
        box.set(facecolor=rgb_colors[box_index])

def get_cortex_ids_to_node_degrees(adjacency_matrix):
    cortex_ids_to_node_degrees = dict()
    atlas_df = get_hcp_atlas_df()
    cortex_ids = atlas_df['Cortex_ID']

    for i in range(NUM_ROIS):
        cortex_id = cortex_ids[i]
        cortex_ids_to_node_degrees[cortex_id] = \
            cortex_ids_to_node_degrees.get(cortex_id, 0) + np.sum(adjacency_matrix[i])
    
    for cortex_id in cortex_ids_to_node_degrees:
        cortex_ids_to_node_degrees[cortex_id] /= CORTEX_IDS_TO_ROI_DIST[cortex_id]
    
    return cortex_ids_to_node_degrees

def viz_node_degree_clusters_for_subject_id(sbj_id, use_super_node=False):
    from utils.data_utils import get_adjacency_matrix_cam_can
    data_path = os.path.join(os.environ['DATAPATH'], CAMCAN_DATSET_STR)
    VIZ_THRESHOLD = 0.36
    adjacency_matrix = get_adjacency_matrix_cam_can(VIZ_THRESHOLD, sbj_id, data_path, use_super_node)
    cortex_ids_to_node_degrees = get_cortex_ids_to_node_degrees(adjacency_matrix)
    
    x = np.arange(len(cortex_ids_to_node_degrees))
    for i, cortex_id in zip(range(len(x)), cortex_ids_to_node_degrees):
        plt.bar(x[i], cortex_ids_to_node_degrees[cortex_id], color=COLORS[cortex_id])
    plt.title(f"ROI Cortical Subnetwork Node Degrees CamCAN Subject {sbj_id}")
    plt.xlabel("Cortical Subnetwork")
    plt.ylabel("Total Cluster Node Degree (Sum of all node degrees within cluster)")
    plt.xticks(x, CORTEX_TO_ABBREVIATION.values(), rotation = 90)
    plt.ylim(0, 120)
    plt.show()

def get_cortices_to_colors():
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    cortices_to_colors = {}

    for i in range(len(COLORS) - 1):
        cortices_to_colors[CORTEX_TO_ABBREVIATION[[*cortex_ids_to_cortices.values()][i]]] = COLORS[i + 1] 
    return cortices_to_colors

def viz_roi_distribution():
    _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
    # Sample data
    # Create a colormap using 'jet'
    cmap = plt.get_cmap('jet')

    # Create a bar plot with Jet colormap
    cortices_to_colors = get_cortices_to_colors()
    number_of_rois = [2, 6, 12, 14, 18, 10, 18, 14, 10, 14, 16, 24, 16, 16, 10, 20, 20, 26, 32, 18, 18, 26]
    x = np.arange(len(number_of_rois))

    # for i, cortex_color in zip(range(len(x)), cortices_to_colors.values()):
    #     plt.bar(x[i], number_of_rois[i], color= cmap(x[i] / len(x)))
    # plt.title("Number of Sorted ROIs per Cortical Subregion")
    # plt.xlabel("Cortex Region")
    # plt.ylabel("Number of ROIs (Left and Right Hemispheres)")
    # plt.xticks(x, sorted_cortex_abbreviations, rotation = 90)

    for i, cortex_id in zip(range(len(x)), cortex_ids_to_cortices):
        plt.bar(x[i], number_of_rois[cortex_id - 1], color=COLORS[cortex_id])
    plt.title("Number of ROIs per Cortical Subregion")
    plt.xlabel("Cortex Region")
    plt.ylabel("Number of ROIs (Left and Right Hemispheres)")
    plt.xticks(x, CORTEX_TO_ABBREVIATION.values(), rotation = 90)

    plt.show()

def plot_thresholds_to_age_prediction_metrics(date, use_edge_fracs = False, model_type="linear"):
    if model_type == "linear":
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    elif model_type == "ridge": 
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    else: 
        raise ValueError("Invalid model type")
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
    plt.figure()
    plt.title(f"Mean Squared Error vs. {x_str}")
    plt.ylim(0, 1500)
    for x_value, mse in zip(x_values, mean_squared_errors):
        plt.scatter(x_value, mse)
    y_values = mean_squared_errors
    hop_len = len(y_values) // 4
    for i in range(len(y_values) // 4):
        plt.plot([x_values[i], x_values[i + hop_len], x_values[i + 2 * hop_len], x_values[i + 3 * hop_len]], 
            [y_values[i], y_values[i + hop_len], y_values[i + 2 * hop_len], y_values[i + 3 * hop_len]])

    plt.xlabel(x_str[ : -1]) # remove s character
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.figure()
    plt.ylim(0, 1)
    plt.title(f"Correlation vs. {x_str}")
    for x_value, correlation in zip(x_values, correlations):
        plt.scatter(x_value, correlation)
    y_values = correlations
    hop_len = len(y_values) // 4
    for i in range(len(y_values) // 4):
        plt.plot([x_values[i], x_values[i + hop_len], x_values[i + 2 * hop_len], x_values[i + 3 * hop_len]], 
            [y_values[i], y_values[i + hop_len], y_values[i + 2 * hop_len], y_values[i + 3 * hop_len]])

    plt.xlabel(x_str[ : -1]) # remove s character
    plt.ylabel("Correlation")
    plt.show()

def plot_thresholds_to_age_prediction_metrics_box_plots(date, use_edge_fracs = False, model_type='linear'):
    if model_type == "linear":
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    elif model_type == "ridge": 
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    else: 
        raise ValueError("Invalid model type")
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
    for y_values in [mean_squared_errors, correlations]:
        plt.figure()
        y_str = "Mean Squared Error" if y_values == mean_squared_errors \
            else "Correlation"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        plt.title(f"{y_str} vs. {x_str}")
        hop_len = len(y_values) // 4
        if use_edge_fracs:
            bp = plt.boxplot([
                [y_values[i + 3 * hop_len] for i in range(len(y_values) // 4)],
                [y_values[i + 2 * hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i] for i in range(len(y_values) // 4)], 
                ], 
                notch=True,
                patch_artist=True)
        else:
            bp = plt.boxplot([
                [y_values[i] for i in range(len(y_values) // 4)], 
                [y_values[i + hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + 2 * hop_len] for i in range(len(y_values) // 4)], 
                [y_values[i + 3 * hop_len] for i in range(len(y_values) // 4)],
                ], 
                notch=True,
                patch_artist=True)
        NUM_DIFF_THRESHOLDS = 4
        rgb_colors = [mcolors.hex2color(COLORS[i]) for i in range(NUM_DIFF_THRESHOLDS)]

        if not use_edge_fracs: rgb_colors = rgb_colors[::-1]
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)
        if use_edge_fracs:
            plt.xticks([1, 2, 3, 4], [0.05, 0.06, 0.07, 0.08])
        else:
            plt.xticks([1, 2, 3, 4], [FIVE_PERCENT_THRESHOLD, SIX_PERCENT_THRESHOLD, SEVEN_PERCENT_THRESHOLD, EIGHT_PERCENT_THRESHOLD][::-1])
        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()

def plot_single_threshold_to_age_prediction_metrics_box_plot(date, use_edge_fracs = False, model_type='linear'):
    
    if model_type == "linear":
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    elif model_type == "ridge": 
        mean_squared_errors, correlations, thresholds = get_age_prediction_metrics_with_thresholds(date, model_type)
    else: 
        raise ValueError("Invalid model type")
    if use_edge_fracs: x_values = [THRESHOLDS_TO_EDGE_FRACTIONS[threshold] for threshold in thresholds]
    else: x_values = thresholds
    x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
    for y_values in [mean_squared_errors, correlations]:
        plt.figure()
        
        y_str = "Mean Squared Error" if y_values == mean_squared_errors \
            else "Correlation"
        x_str = 'Thresholds' if not use_edge_fracs else 'Edge Fractions'
        if y_str == "Mean Squared Error": plt.ylim(0, 1200)
        elif y_str == "Correlation": plt.ylim(-1, 1)
        else: raise AssertionError(f"Invalid y_str value : {y_str}")
        plt.title(f"{y_str} vs. {x_str}")
        bp = plt.boxplot(y_values, notch=True, patch_artist=True)

        NUM_DIFF_THRESHOLDS = 4
        rgb_colors = [mcolors.hex2color(COLORS[2]) for i in range(NUM_DIFF_THRESHOLDS)]

        if not use_edge_fracs: rgb_colors = rgb_colors[::-1]
        for box, color in zip(bp['boxes'], rgb_colors):
            box.set(facecolor=color)
        if use_edge_fracs:
            plt.xticks([1], [0.05])
        else:
            plt.xticks([1], [FIVE_PERCENT_THRESHOLD])
        plt.xlabel(x_str[ : -1]) # remove s character
        plt.ylabel(y_str)
        plt.show()

def plot_roi_hyperbolic_radii_distribution_box_plots(date):
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    
    NUMBER_OF_ITERATIONS = 4
    hyperbolic_radii_vectors = [get_roi_hyperbolic_radii_list(embeddings_df) for embeddings_df in embeddings_df_list ]
    hyperbolic_radii_distribution_per_roi = np.array(hyperbolic_radii_vectors)   
    for iter_index in range(NUMBER_OF_ITERATIONS):
        plt.figure(figsize=(20, 3))
        hyperbolic_radii_distribution_per_roi_section = hyperbolic_radii_distribution_per_roi[ : , iter_index * (NUM_ROIS // 4) : (iter_index + 1) * (NUM_ROIS // 4)]
        bp = plt.boxplot(hyperbolic_radii_distribution_per_roi_section, 
                        patch_artist=True, 
                        notch=False)
        cmap = plt.cm.jet
        
        for box, index in zip(bp['boxes'], range(NUM_ROIS // 4)):
            box.set(facecolor=cmap(((iter_index * NUM_ROIS // 4) + index) / NUM_ROIS))
        plt.title(f"ROI Hyperbolic Radii Distribution {iter_index * NUM_ROIS // 4} : {(iter_index + 1) * NUM_ROIS // 4}")
        plt.ylabel("Hyperbolic Radius")
        plt.xticks(ticks=range(NUM_ROIS // 4), labels=range(iter_index * (NUM_ROIS // 4) , (iter_index + 1) * (NUM_ROIS // 4)), rotation=90)
        plt.xlabel("ROI Index")


def viz_roi_hyperbolic_radii_vs_thickness(date):
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    thick_features = get_thick_features(100, data_path)
    hyperbolic_radii_vectors = [get_roi_hyperbolic_radii_list(embeddings_df) for embeddings_df in embeddings_df_list ]
    hyperbolic_radii_distribution_per_roi = np.array(hyperbolic_radii_vectors)   
    
    average_hyperbolic_radii = np.sum(hyperbolic_radii_distribution_per_roi, axis=0)
    min_max_normalize_thick_features = min_max_normalize(thick_features)
    min_max_normalize_average_hyperbolic_radii = min_max_normalize(average_hyperbolic_radii)
    plt.title(f"ROI Hyperbolic Radii and Thickness")
    plt.ylabel("Hyperbolic Radius")
    plt.xlabel("ROI Index")

    plt.plot(min_max_normalize_average_hyperbolic_radii, label="ROI Hyperbolic Radius")
    plt.plot(min_max_normalize_thick_features, label="ROI Cortical Thickness")
    plt.legend()

def viz_roi_hyperbolic_radii_vs_myelination(date):
    embeddings_df_list = get_embeddings_df_list_all_splits(date) # Only gets test embeddings: NOT ANYMORE!
    data_path = os.path.join(os.environ['DATAPATH'], "cam_can_multiple")
    myelin_features = get_myelin_features(100, data_path)
    hyperbolic_radii_vectors = [get_roi_hyperbolic_radii_list(embeddings_df) for embeddings_df in embeddings_df_list ]
    hyperbolic_radii_distribution_per_roi = np.array(hyperbolic_radii_vectors)   
    
    average_hyperbolic_radii = np.sum(hyperbolic_radii_distribution_per_roi, axis=0)
    min_max_normalize_myelin_features = min_max_normalize(myelin_features)
    min_max_normalize_average_hyperbolic_radii = min_max_normalize(average_hyperbolic_radii)
    plt.title(f"ROI Hyperbolic Radii and Myelination")
    plt.ylabel("Hyperbolic Radius")
    plt.xlabel("ROI Index")

    plt.plot(min_max_normalize_average_hyperbolic_radii, label="ROI Hyperbolic Radius")
    plt.plot(min_max_normalize_myelin_features, label="ROI Myelination")
    plt.legend()

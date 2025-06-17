import os
import pandas as pd
import torch
import numpy as np
from utils.constants_utils import NUM_ROIS, NUM_MEG_COLE_ROIS, COLORS, DECADE_POSITIONS_STR_LIST, NUM_DECADES, DATAPATH, NUM_SUBNETS

os.environ['DATAPATH'] = DATAPATH
def to_poincare(x, c):
    K = 1. / c
    sqrtK = K ** 0.5
    d = x.size(-1) - 1
#     return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
    return sqrtK * x.narrow(-1, 1, d) / (x[0] + sqrtK)
def get_subnetworks():
    """
    Returns set with subnetwork indices (0-indexed)
    """
    hcp_atlas_df = get_hcp_atlas_df()
    cortex_ids = hcp_atlas_df['Cortex_ID']
    subnetworks = [set() for i in range(NUM_SUBNETS)]
    for i in range(NUM_ROIS):
        subnetworks[cortex_ids[i] - 1].add(i)
    return subnetworks
def get_subnetworks_left_right():
    """
    Returns set with left and right subnetwork indices (0-indexed)
    """
    hcp_atlas_df = get_hcp_atlas_df()
    cortex_ids = hcp_atlas_df['Cortex_ID']
    subnetworks_left = [set() for i in range(NUM_SUBNETS)]
    subnetworks_right = [set() for i in range(NUM_SUBNETS)]
    for i in range(NUM_ROIS):
        if hcp_atlas_df['LR'][i] == 'L': subnetworks_left[cortex_ids[i] - 1].add(i)
        if hcp_atlas_df['LR'][i] == 'R': subnetworks_right[cortex_ids[i] - 1].add(i)
    return subnetworks_left, subnetworks_right



def get_hcp_atlas_df():
    datapath = os.environ['DATAPATH']
    atlas_path = os.path.join(os.path.join(datapath, "cam_can_avg_new"), "HCP-MMP1_UniqueRegionList.csv")
    hcp_atlas_df = pd.read_csv(atlas_path)
    cortices = hcp_atlas_df['cortex']
    region_indices_to_cortices = {index : cortex for index, cortex in enumerate(cortices)}
    color_index = -1
    seen_cortices = set()
    region_indices_to_colors = {}

    for index in region_indices_to_cortices:
        cortex = region_indices_to_cortices[index]
        if cortex not in seen_cortices:
            color_index += 1
            region_indices_to_colors[index] = COLORS[color_index]
            seen_cortices.add(region_indices_to_cortices[index])    
        else:
            region_indices_to_colors[index] = COLORS[color_index]
        
    return hcp_atlas_df
def get_cortices_and_cortex_ids_to_cortices():
    hcp_atlas_df = get_hcp_atlas_df()
    cortices = hcp_atlas_df['cortex']
    cortex_ids = hcp_atlas_df['Cortex_ID']
    cortex_ids_to_cortices = {cortex_ids[i] : cortices[i] for i in range(NUM_ROIS)}
    
    return cortices, cortex_ids_to_cortices

def get_embeddings_df(hyperboloid_embeddings):
    """
    Returns: 
    embeddings_df with columns 
        x : x coordinate in Poincare Disk :float, 
        y : y coordinate in Poincare Disk :float, 
        label : cortex id number :int [1 - 22], 
        id : ROI id number :int [0 - 359],
        LR : left or right hemisphere embedding symbolized by 'L' or 'R' :str
    """
    hcp_atlas_df = get_hcp_atlas_df()
    
    region_indices_to_cortex_ids = {i : hcp_atlas_df['Cortex_ID'][i] for i in range(NUM_ROIS)}
    c = 1.
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    
    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(NUM_ROIS)], 
                                        'y': [poincare_embeddings[i][1] for i in range(NUM_ROIS)],
                                        'label': [region_indices_to_cortex_ids[i] for i in range(NUM_ROIS)],
                                        'id': [i for i in range(NUM_ROIS)],
                                        'LR': hcp_atlas_df['LR']}
                                )
    return embeddings_df   

def get_meg_embeddings_df(hyperboloid_embeddings):
    # aal_atlas_df = get_meg_atlas_df()
    
    # region_indices_to_cortex_ids = {i : aal_atlas_df['Cortex_ID'][i] for i in range(NUM_MEG_COLE_ROIS)}
    c = 1.
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    # TODO: Add proper LR and Label check from MEG atlas csv file
    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(NUM_MEG_COLE_ROIS)], 
                                        'y': [poincare_embeddings[i][1] for i in range(NUM_MEG_COLE_ROIS)],
                                        'label': [i // 4 + 1 for i in range(NUM_MEG_COLE_ROIS)],
                                        'id': [i for i in range(NUM_MEG_COLE_ROIS)],
                                        'LR': ["L" if i % 4 <= 1 else "R" for i in range(NUM_MEG_COLE_ROIS)]}
                                )
    return embeddings_df   

def get_tree_embeddings_df(hyperboloid_embeddings):
    c = 1.
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
    
    embeddings_df = pd.DataFrame({'x': [poincare_embeddings[i][0] for i in range(len(poincare_embeddings))], 
                                        'y': [poincare_embeddings[i][1] for i in range(len(poincare_embeddings))],
                                        'label': [int(np.log2(index + 1)) for index in range(len(poincare_embeddings))],
                                        'id': [i for i in range(len(poincare_embeddings))],
                                        }
                                )
    return embeddings_df   

def scale_embeddings_df_to_poincare_disk(embeddings_df):
    eps = 1e-2
    embeddings_df['r'] = torch.sqrt(torch.Tensor(embeddings_df['x']) ** 2 + torch.Tensor(embeddings_df['y']) ** 2)
    max_radius = np.max(embeddings_df.r)
    embeddings_df['x'] /= (max_radius + eps)
    embeddings_df['y'] /= (max_radius + eps)
    embeddings_df['r'] /= (max_radius + eps)
    return embeddings_df

def get_embeddings_df_list(date):
    """
    Only gets embeddings df from test embeddings
    1. Gather all previous runs of embeddings in the date file directory from all the log numbers in the directory, 
    2. Access the respective numpy embeddings file, 
    3. Convert the numpy file into a dataframe, 
    4. Append dataframe to a list of embeddings dataframes 
    """
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    log_num_dirs = [os.path.join(date_dir, log_num) for log_num in os.listdir(date_dir)]
    embeddings_df_list = []
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    for log_num_dir in log_num_dirs:
        for root, dirs, files in os.walk(log_num_dir):
            for file in files:
                if file.endswith(".npy"):
                    if 'test' not in file: continue 
                    embeddings = np.load(os.path.join(root, file))
                    embeddings_df = get_embeddings_df(embeddings)
                    NUM_INDICES_TO_SKIP_NUMPY_SUFFIX = 4
                    DATA_SPLIT_FILE_INDEX = 1
                    sbj_num = int(file.split("_")[-1][:-NUM_INDICES_TO_SKIP_NUMPY_SUFFIX])
                    data_split = file.split("_")[DATA_SPLIT_FILE_INDEX] # train, val, test split
                    embeddings_df['sbj_num'] = sbj_num
                    embeddings_df['age_label'] = age_labels[sbj_num]
                    embeddings_df_list.append(embeddings_df)
    return embeddings_df_list

def get_embeddings_df_list_all_splits(date):
    """
    Acquires gets embeddings df from ALL SPLITS
    1. Gather all previous runs of embeddings in the date file directory from all the log numbers in the directory, 
    2. Access the respective numpy embeddings file, 
    3. Convert the numpy file into a dataframe, 
    4. Append dataframe to a list of embeddings dataframes 
    """
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    log_num_dirs = [os.path.join(date_dir, log_num) for log_num in os.listdir(date_dir)]
    embeddings_df_list = []
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
    for log_num_dir in log_num_dirs:
        for root, dirs, files in os.walk(log_num_dir):
            for file in files:
                if file.endswith(".npy"):
                    if 'best' in file: continue 
                    embeddings = np.load(os.path.join(root, file))
                    embeddings_df = get_embeddings_df(embeddings)
                    NUM_INDICES_TO_SKIP_NUMPY_SUFFIX = 4
                    DATA_SPLIT_FILE_INDEX = 1
                    sbj_num = int(file.split("_")[-1][:-NUM_INDICES_TO_SKIP_NUMPY_SUFFIX])
                    data_split = file.split("_")[DATA_SPLIT_FILE_INDEX] # train, val, test split
                    embeddings_df['sbj_num'] = sbj_num
                    embeddings_df['age_label'] = age_labels[sbj_num]
                    embeddings_df_list.append(embeddings_df)
    return embeddings_df_list

def acquire_ordered_embeddings_df_list_all_splits(date):
    """
    Acquires gets embeddings df from ALL SPLITS
    1. Gather all previous runs of embeddings in the date file directory from all the log numbers in the directory, 
    2. Access the respective numpy embeddings file, 
    3. Convert the numpy file into a dataframe, 
    4. Append dataframe to a list of embeddings dataframes 
    """
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    log_num_dirs = [os.path.join(date_dir, log_num) for log_num in os.listdir(date_dir)]
    embeddings_df_list = []
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))

    for log_num_dir in log_num_dirs:
        embeddings_dir = os.path.join(log_num_dir, "embeddings")
        for embeddings_file in os.listdir(embeddings_dir):
            print("EMBEDDINGS FILE", embeddings_file)
            if embeddings_file.endswith(".npy"):
                embeddings = np.load(os.path.join(embeddings_dir, embeddings_file))
                embeddings_df = get_embeddings_df(embeddings)
                NUM_INDICES_TO_SKIP_NUMPY_SUFFIX = 4
                DATA_SPLIT_FILE_INDEX = 1
                sbj_num = int(embeddings_file.split("_")[-1][:-NUM_INDICES_TO_SKIP_NUMPY_SUFFIX])
                data_split = embeddings_file.split("_")[DATA_SPLIT_FILE_INDEX] # train, val, test split
                embeddings_df['sbj_num'] = sbj_num
                embeddings_df['age_label'] = age_labels[sbj_num]
                embeddings_df_list.append(embeddings_df)
    return embeddings_df_list


def get_embeddings_df_list_with_age_labels(date):
    """
    Acquires gets embeddings df from ALL SPLITS
    1. Gather all previous runs of embeddings in the date file directory from all the log numbers in the directory, 
    2. Access the respective numpy embeddings file, 
    3. Convert the numpy file into a dataframe, 
    4. Append dataframe to a list of embeddings dataframes 
    """
    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    log_num_dirs = [os.path.join(date_dir, log_num) for log_num in os.listdir(date_dir)]
    embeddings_df_list = []
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))

    for log_num_dir in log_num_dirs:
        for root, dirs, files in os.walk(log_num_dir):
            for file in files:
                if file.endswith(".npy"):
                    # if 'test' not in file: continue 
                    if 'best' in file: continue 
                    embeddings = np.load(os.path.join(root, file))
                    
                    NUM_INDICES_TO_SKIP_NUMPY_SUFFIX = 4
                    DATA_SPLIT_FILE_INDEX = 1
                    
                    sbj_num = int(file.split("_")[-1][:-NUM_INDICES_TO_SKIP_NUMPY_SUFFIX])
                    data_split = file.split("_")[DATA_SPLIT_FILE_INDEX] # train, val, test split
                    embeddings_df = get_embeddings_df(embeddings)
                    # embeddings_df['age_label'] = graph_data_dicts[sbj_num]['age_label']
                    embeddings_df['sbj_num'] = sbj_num
                    embeddings_df['age_label'] = age_labels[sbj_num]
                    embeddings_df_list.append(embeddings_df)
    return embeddings_df_list


def get_embeddings_df_list_by_decade(date, nth_index):
    """
    1. Gather all previous runs of embeddings in the date file directory from all the log numbers in the directory, 
    2. Access the respective numpy embeddings file, 
    3. Convert the numpy file into a dataframe, 
    4. Append dataframe to a list of embeddings dataframes BY DECADE!!!!!!!!
    """
    embeddings_df_list_decade = []
    import logging
    nth_str = DECADE_POSITIONS_STR_LIST[nth_index]
    num_sbjs = 70
    logging.info(f"Using Only {nth_str} {num_sbjs} Subjects")
    lower_bound = num_sbjs * nth_index
    upper_bound = lower_bound + num_sbjs
    indices_from_decade = [index for index in range(lower_bound, upper_bound)]
    if nth_index == NUM_DECADES - 1: upper_bound = 587 # [490, 587) -> 97

    date_dir = os.path.join(os.getcwd(), 'logs', 'lp', date)
    log_num_dirs = [os.path.join(date_dir, log_num) for log_num in os.listdir(date_dir)]
    age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))

    for log_num_dir in log_num_dirs:
        for root, dirs, files in os.walk(log_num_dir):
            for file in files:
                if file.endswith(".npy"):
                    # if 'test' not in file: continue 
                    if 'best' in file: continue 
                    embeddings = np.load(os.path.join(root, file))
                    
                    NUM_INDICES_TO_SKIP_NUMPY_SUFFIX = 4
                    DATA_SPLIT_FILE_INDEX = 1
                    
                    sbj_num = int(file.split("_")[-1][:-NUM_INDICES_TO_SKIP_NUMPY_SUFFIX])
                    data_split = file.split("_")[DATA_SPLIT_FILE_INDEX] # train, val, test split

                    if sbj_num not in indices_from_decade: continue
                    print("THESE ARE THE SBJ NUMS BEING USED", sbj_num)
                    embeddings_df = get_embeddings_df(embeddings)
                    embeddings_df['age_label'] = age_labels[sbj_num]
                    embeddings_df_list_decade.append(embeddings_df)
    return embeddings_df_list_decade

# Visualization script for VCI embeddings
# This script visualizes hyperboloid embeddings of a VCI individual in the Poincaré disk model.
# It reads embeddings from a specified path, converts them to Poincaré coordinates,
# and plots them with different markers and colors based on hemisphere and base region labels.


from visualization import *
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

# Function to convert hyperboloid embeddings to Poincaré coordinates and visualize them
def vci_viz_embeddings(embeddings_path,
                       labels,
                       use_save=False,
                       prefix=""):

    embeddings_name = embeddings_path.split('\\')[-1]
    if 'log' in embeddings_name:
        return

    # Load and convert embeddings
    hyperboloid_embeddings = np.load(embeddings_path)
    c = 1.0
    torch_embeddings = torch.from_numpy(hyperboloid_embeddings)
    poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]

    # Step 1: Strip hemisphere to get base region names
    base_labels = []
    hemispheres = []
    for label in labels:
        if label.startswith("Left "):
            base_labels.append(label[5:])
            hemispheres.append("Left")
        elif label.startswith("Right "):
            base_labels.append(label[6:])
            hemispheres.append("Right")
        else:
            base_labels.append(label)
            hemispheres.append("Unknown")

    # Step 2: Assign color per base label
    unique_base_labels = sorted(set(base_labels))
    cmap = get_cmap("tab20")
    label_to_color = {
        region: to_hex(cmap(i % 20))
        for i, region in enumerate(unique_base_labels)
    }

    # Step 3: Plot setup
    fig, ax = plt.subplots()
    circ = plt.Circle((0, 0), radius=1, edgecolor='black', facecolor='None', linewidth=3, alpha=0.5)
    ax.add_patch(circ)
    ax.set_aspect(0.9)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    # ax.set_title(f"{embeddings_name[:-4]}")
    ax.set_title("Hyperboloid Embeddings of a VCI Individual")

    # Step 4: Plot each point
    for point, base_label, hemisphere in zip(poincare_embeddings, base_labels, hemispheres):
        x, y = point[0].item(), point[1].item()
        color = label_to_color.get(base_label, 'gray')
        marker = '^' if hemisphere == "Left" else 'o' if hemisphere == "Right" else 'x'
        ax.plot(x, y, marker=marker, color=color, markersize=5)

    # Step 5: Construct custom legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Left Hemisphere',
               markerfacecolor='black', markersize=6),
        Line2D([0], [0], marker='o', color='w', label='Right Hemisphere',
               markerfacecolor='black', markersize=6)
    ]
    for region in unique_base_labels:
        legend_elements.append(
            Line2D([0], [0], marker='s', color='w', label=region,
                   markerfacecolor=label_to_color[region], markersize=6)
        )

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0., fontsize=7, ncol=1)

    # Step 6: Save or show
    if use_save:
        save_path = f"{prefix}{embeddings_name[:-4]}_viz.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


# Read labels from the text file
labels_path = os.path.join(os.getcwd(), 'data', 'vci', 'vci_labels.txt')
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Visualize embeddings
log_dir = os.path.join("F:\\MYPROJECTS17\\project_RF1_VCI_networkanalysis\\fhnn_vci_public\\logs\\lp\\2025_6_16\\12\\embeddings")
embeddings_path = os.path.join(log_dir, "embeddings_train_2.npy")
vci_viz_embeddings(embeddings_path, labels)






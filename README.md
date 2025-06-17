# Hyperbolic embeddings of MEG brain networks using FHNN 

This project computes the hyperbolic embeddings of MEG brain networks using the Fully Hyperbolic Neural Network model. See references below:

*Weize Chen, Xu Han, Yankai Lin, Hexu Zhao, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. 2022. Fully Hyperbolic Neural Networks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5672–5686, Dublin, Ireland. Association for Computational Linguistics.*

*Ramirez, H., Tabarelli, D., Brancaccio, A., Belardinelli, P., Marsh, E.B., Funke, M., Mosher, J.C., Maestu, F., Xu, M. and Pantazis, D., 2025. Fully hyperbolic neural networks: A novel approach to studying aging trajectories. IEEE Journal of Biomedical and Health Informatics.*


# Project organization for network embedding
Source code based on [HGCN](https://github.com/HazyResearch/hgcn) and [FHNN](https://github.com/chenweize1998/fully-hyperbolic-nn) repositories. File structure for FHNN source code:

```
📦gcn
 ┣ 📂data
 ┣ 📂layers
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜att_layers.py
 ┃ ┣ 📜hyp_layers.py    # Defines Lorentz Graph Convolutional Layer
 ┃ ┗ 📜layers.py
 ┣ 📂manifolds
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜euclidean.py
 ┃ ┣ 📜hyperboloid.py
 ┃ ┣ 📜lmath.py         # Math related to our manifold
 ┃ ┣ 📜lorentz.py       # Our manifold
 ┃ ┣ 📜poincare.py
 ┃ ┗ 📜utils.py
 ┣ 📂models
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_models.py
 ┃ ┣ 📜decoders.py      # Include FHNN decoder
 ┃ ┗ 📜encoders.py      # Include FHNN encoder
 ┣ 📂optim
 ┣ 📂utils
 ┣ 📜step1_import_plv_matrices.py       # Imports MEG PLV matrices from *.mat files
 ┣ 📜step2_train_graph_iteration.py     # Computes hyperbolic embeddings
 ┣ 📜step3_vci_visualization.py         # Visualizes hyperbolic embeddings
 
 ```

## Training FHNN model using Graph Iteration
Arguments passed to program:

`--task` Specifies task. Can be [lp], lp denotes link prediction.

`--act` Activation function (e.g. relu, tanh). Use None for no activation.

`--dataset` Specifies dataset.

`--lr` Specifies learning rate.

`--dim` Specifies dimension of embeddings.

`--num-layers` Specifies number of layers.

`--bias` To enable the bias, set it to 1.

`--dropout` Specifies dropout rate.

`--weight-decay` Specifies weight decay value.

`--log-freq` Interval for logging.

For other arguments, see `config.py`

In a Jupyter notebook, you can run: 

! python train_graph_iteration.py \
    --task lp \
    --act None \
    --dataset vci \
    --model HyboNet \
    --lr 0.05 \
    --dim 3 \
    --num-layers 2 \
    --bias 1 \
    --dropout 0.25 \
    --weight-decay 1e-3 \
    --manifold Lorentz \
    --log-freq 5 \
    --cuda -1 \
    --patience 50 \
    --grad-clip 0.1 \
    --seed 1234 \
    --save 1

In a terminal, run as follows: 

```bash
python step2_train_graph_iteration.py --task lp --act None --dataset vci --model HyboNet --lr 0.05 --dim 3 --num-layers 2 --bias 1 --dropout 0.25 --weight-decay 1e-3 --manifold Lorentz --log-freq 5 --cuda -1 --patience 50 --grad-clip 0.1 --seed 1234 --save 1
```

## Project installation

If you use conda, you can recreate the environment with:

```bash
conda env create -f environment.yml -n <env-name>
conda activate <env-name>
```

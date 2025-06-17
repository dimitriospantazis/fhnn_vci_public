# Age Prediction through hyperbolic radius extraction from hyperbolic embeddings of MEG brain networks using FHNN 
Age Prediction using [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686) 

```
@article{chen2021fully,
  title={Fully Hyperbolic Neural Networks},
  author={Chen, Weize and Han, Xu and Lin, Yankai and Zhao, Hexu and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2105.14686},
  year={2021}
}
{"mode":"full","isActive":false}
```

# Codes for Network Embedding
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
 ```

## 1. Training FHNN model using Graph Iteration
Arguments passed to program:

`--task` Specifies task. Can be [lp], lp denotes link prediction.

`--dataset` Specifies dataset.

`--lr` Specifies learning rate.

`--dim` Specifies dimension of embeddings.

`--num-layers` Specifies number of layers.

`--bias` To enable the bias, set it to 1.

`--dropout` Specifies dropout rate.

`--weight-decay` Specifies weight decay value.

`--log-freq` Interval for logging.

For other arguments, see `config.py`

In a Jupyter notebook, you can run an an example-run as follows: 

! python train_graph_iteration.py \
    --task lp \
    --act None \
    --dataset cam_can_multiple\
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
    --patience 500 \
    --grad-clip 0.1 \
    --seed 1234 \
    --save 1

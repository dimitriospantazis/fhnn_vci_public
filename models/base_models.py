"""Base model class."""

from copy import deepcopy
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperbolic_clustering.utils.utils import poincare_dist
from layers.layers import FermiDiracDecoder 
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1, MarginLoss

from models.centroid_regression_model import get_centroid_regression_distance # NOTE: Added for HGNN centroid regression
from utils.constants_utils import NUM_ROIS
import os
NUM_CENTROIDS = 7
EMBEDDING_DIMENSION = 3
use_centroid_loss = False

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:  #if curvature is provided (not trainable)
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.])) #if curvature is trainable
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        if self.manifold_name == 'Lorentz':
            correct = output.gather(1, data['labels'][idx].unsqueeze(-1))
            loss = F.relu(self.margin - correct + output).mean()
        else:
            loss = F.cross_entropy(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.args = args
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(args.margin)
        # self.loss = MarginLoss(args.margin) + nn.MSELoss(reduction='sum')
        if use_centroid_loss:
            self.graph_prediction_loss_function = nn.MSELoss()
            self.centroid_embedding = torch.nn.Embedding(
                NUM_CENTROIDS, EMBEDDING_DIMENSION,
                sparse=False,
                scale_grad_by_freq=False
            )    
            self.output_linear = nn.Linear(NUM_CENTROIDS, 1) # 1 since regression

        # Cole added
        self.is_inductive = True # Makes sense since Cole called the file train_inductive.py
        self.epoch_stats = {
            'prefix': 'start',
            'epoch': -1,
            'loss': 0,
            'roc': 0,
            'ap': 0,
            'acc': 0,
            'num_correct': 0, ## add to acc_f1 funct
            'num_true': 0,
            'num_false': 0,
            'num_graphs': 0,
            'num_total': 0,
            'num_updates':0
        }
        logging.getLogger().setLevel(logging.INFO)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # dist = self.manifold.dist(emb_in, emb_out)
        # TODO: Restore to non-FD FHNN with -sqdist
        # Fermi Dirac Decoder Probabilities from HGCN
        # TODO: See how -sqdist does now that thresholds have been increased.... still think that FD HGCN self.dc.forward(sqdist) is more correct
        return self.dc.forward(sqdist)
    
    def get_avg_hyperbolic_radius(self, embeddings):
        origin = torch.Tensor([1, 0, 0]) # .to(self.args.device)
        avg_radius = 0
        for embedding_coords in embeddings:
            hyperbolic_radius = self.manifold.sqdist(embedding_coords, origin)
            avg_radius += hyperbolic_radius
        avg_radius /= len(embeddings)
        return avg_radius
    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        # How does system balance number of positive edges from number of negative edges? SAMPLES THEM!!!
        preds = torch.stack([pos_scores, neg_scores], dim=-1)
        
        # preds = torch.stack([exp_pos_scores, exp_neg_scores], dim=-1)
        loss = self.loss(preds)
        
        # avg_radius = self.get_avg_hyperbolic_radius(embeddings)
        # loss += 100 * avg_radius
        
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        prediction_threshold = 0.5 # TODO: Check if 0.5 is the appropriate threshold
        preds_binary = np.where(np.array(preds) > prediction_threshold, 1, 0)
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        metrics = {'loss': loss, 'roc': roc, 'ap': ap, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
        return metrics

    def compute_metrics_multiple(self, embeddings_list, graph_data_dicts, split):
        # TODO: See if can add average radius regularization here as well (do not want data to clump up at origin)
        edges_full=[]
        edges_false_full=[]
        pos_probs_full=[]
        neg_probs_full=[]
        pos_scores_full=[]
        neg_scores_full=[]
        # LOSS_FREQ = 1
        # cohesion_loss = 0
        centroid_loss = torch.Tensor([0])
        
        # Tensorifying embeddings list
        # embeddings_tensor = np.array(embeddings_list)
        embeddings_tensor = [torch.Tensor(embeddings) for embeddings in embeddings_list]
        #embeddings_tensor = torch.Tensor(embeddings_tensor)
        embeddings_tensor = torch.Tensor(embeddings_tensor[0])
       
        if use_centroid_loss:
            mask = torch.ones(self.centroid_embedding.num_embeddings, dtype=torch.float32)
            for i in range(len(embeddings_list)):
                centroid_dists = get_centroid_regression_distance(embeddings_tensor[i], 
                                                    mask, 
                                                    self.centroid_embedding, 
                                                    self.manifold)
                predicted_age = self.output_linear(centroid_dists)[0]
                centroid_loss += self.graph_prediction_loss_function(predicted_age, age_label)
        # TODO: Should normalize and rebalance centroid GP loss with LP loss so that it is even 
        for i in range(len(embeddings_list)):
            embeddings = embeddings_list[i]
            graph_data_dict = graph_data_dicts[i]
            edges, edges_false = self.get_edges(embeddings, graph_data_dict, split)

            pos_scores = self.decode(embeddings, edges)
            neg_scores = self.decode(embeddings, edges_false)

            if len(pos_scores.shape) > 1:
                assert pos_scores.shape[1] == 1
                pos_scores = pos_scores[:, 0]
                neg_scores = neg_scores[:, 0]
            # adj_prob = graph_data['adj_prob']

            if self.args.dataset == 'cam_can_multiple' and self.args.use_thickness:
                plv_matrix = graph_data_dict['features'][:, 2:]
            else:
                plv_matrix = graph_data_dict['features']
            plv_matrix_numpy = plv_matrix.clone().detach().cpu().numpy()
            
            # Save time for margin loss calculations since PLV probabilities not needed for margin loss
            if not self.args.use_margin_loss:
                # TODO: Consider simply min-max normalizing the matrix instead of running Cole's function (Cole uses a 95% min-max normalization)
                # adj_prob = self.get_adj_prob(plv_matrix_numpy, is_stretch_sigmoid=False)
                
                from utils.data_utils import min_max_normalize
                adj_prob = min_max_normalize(plv_matrix_numpy)
                neg_probs = self.true_probs(adj_prob, edges_false)
                pos_probs = self.true_probs(adj_prob, edges)

                pos_probs_tensor = torch.Tensor(pos_probs)
                neg_probs_tensor = torch.Tensor(neg_probs)
                
            else:
                pos_probs_tensor = torch.Tensor([0])
                neg_probs_tensor = torch.Tensor([0])
            
            pos_scores_tensor = torch.Tensor(pos_scores)
            neg_scores_tensor = torch.Tensor(neg_scores)
            
            edges_full.append(edges)
            edges_false_full.append(edges_false)
            pos_probs_full.append(pos_probs_tensor)
            neg_probs_full.append(neg_probs_tensor)
            pos_scores_full.append(pos_scores_tensor)
            neg_scores_full.append(neg_scores_tensor)

            # exp_pos_scores_tensor = torch.exp(pos_scores_tensor)
            # exp_neg_scores_tensor = torch.exp(neg_scores_tensor)
            # pos_scores_full.append(exp_pos_scores_tensor)
            # neg_scores_full.append(exp_neg_scores_tensor)
                        
            # avg_radius = self.get_avg_hyperbolic_radius(embeddings)
            
            # Penalize Clumping Up of Embeddings                
            # Cohesion calculation expensive, so only do it every so often
            # is_cohesion_loss_iter = np.random.binomial(1, 1 / LOSS_FREQ)
            # if is_cohesion_loss_iter:
            #     cohesion_loss += 1000 * self.get_distance_between_embeddings(embeddings)
            
        edges_comb = torch.cat(edges_full)
        edges_false_comb = torch.cat(edges_false_full)
        pos_probs_comb = torch.cat(pos_probs_full)
        neg_probs_comb = torch.cat(neg_probs_full)
        pos_scores_comb = torch.cat(pos_scores_full)
        neg_scores_comb = torch.cat(neg_scores_full)
        
        
        ### edges,edges_false only used for their length, so don't matter
        metrics = self.loss_handler(edges_comb,
                                    edges_false_comb,
                                    pos_probs_comb,
                                    neg_probs_comb,
                                    pos_scores_comb,
                                    neg_scores_comb,
                                    num_graphs = len(embeddings_list)
                                    )
        if use_centroid_loss:
            
            metrics['loss'] = torch.Tensor([metrics['loss']])
            metrics['loss'].view((1, 1))
            metrics['loss'] += centroid_loss
            # TODO: Test how only GP without LP loss changes embeddings, currently seems to not learn properly.
            # It is possible that the autograd is getting messed up from recasting the torch tensor.
        return metrics
    
    def compute_metrics_multiple_for_embeddings_dict(self, index_to_embeddings_dict, graph_data_dicts, split):
        # TODO: See if can add average radius regularization here as well (do not want data to clump up at origin)
        edges_full= []
        edges_false_full = []
        pos_probs_full = []
        neg_probs_full = []
        pos_scores_full = []
        neg_scores_full = []
        # Tensorifying embeddings list
        # embeddings_age_labels = torch.Tensor([age_labels[graph_data_dict['index']] for graph_data_dict in graph_data_dicts])
        # embeddings_tensor = torch.Tensor([index_to_embeddings_dict[index] for index in index_to_embeddings_dict])
        # embeddings_tensor = [torch.Tensor(embeddings) for embeddings in embeddings_list]
        # embeddings_tensor = torch.Tensor(embeddings_tensor)
        centroid_loss = torch.Tensor([0])
        if use_centroid_loss:
            mask = torch.ones(self.centroid_embedding.num_embeddings, dtype=torch.float32)
            for index in index_to_embeddings_dict:
                embeddings = index_to_embeddings_dict[index]
                age_label = torch.Tensor([age_labels[index]])
                centroid_dists = get_centroid_regression_distance(embeddings, 
                                                    mask, 
                                                    self.centroid_embedding, 
                                                    self.manifold)
                predicted_age = self.output_linear(centroid_dists).view((1, 1))
                age_label = age_label.view((1, 1))
                centroid_loss += self.graph_prediction_loss_function(predicted_age, age_label)
        # TODO: Should normalize and rebalance centroid GP loss with LP loss so that it is even

        for embeddings_index in index_to_embeddings_dict:
            embeddings = index_to_embeddings_dict[embeddings_index]
            # Find the corresponding graph_data_dict in the list of graph_data_dicts
            for graph_data_dict in graph_data_dicts:
                if graph_data_dict['index'] == embeddings_index:
                    break
            # graph_data_dict = graph_data_dicts[embeddings_index]

            edges, edges_false = self.get_edges(embeddings, graph_data_dict, split)

            pos_scores = self.decode(embeddings, edges)
            neg_scores = self.decode(embeddings, edges_false)

            if len(pos_scores.shape) > 1:
                assert pos_scores.shape[1] == 1
                pos_scores = pos_scores[:, 0]
                neg_scores = neg_scores[:, 0]
            # adj_prob = graph_data['adj_prob']
            
            if self.args.dataset == 'cam_can_multiple' and self.args.use_thickness:
                plv_matrix = graph_data_dict['features'][:, 2:]
            else:
                plv_matrix = graph_data_dict['features']
            plv_matrix_numpy = plv_matrix.clone().detach().cpu().numpy()
            
            # Save time for margin loss calculations since PLV probabilities not needed for margin loss
            if not self.args.use_margin_loss:
                
                from utils.data_utils import min_max_normalize
                adj_prob = min_max_normalize(plv_matrix_numpy)
                neg_probs = self.true_probs(adj_prob, edges_false)
                pos_probs = self.true_probs(adj_prob, edges)

                pos_probs_tensor = torch.Tensor(pos_probs)
                neg_probs_tensor = torch.Tensor(neg_probs)
                
            else:
                pos_probs_tensor = torch.Tensor([0])
                neg_probs_tensor = torch.Tensor([0])
            
            pos_scores_tensor = torch.Tensor(pos_scores)
            neg_scores_tensor = torch.Tensor(neg_scores)
            
            edges_full.append(edges)
            edges_false_full.append(edges_false)
            pos_probs_full.append(pos_probs_tensor)
            neg_probs_full.append(neg_probs_tensor)
            pos_scores_full.append(pos_scores_tensor)
            neg_scores_full.append(neg_scores_tensor)

        edges_comb = torch.cat(edges_full)
        edges_false_comb = torch.cat(edges_false_full)
        pos_probs_comb = torch.cat(pos_probs_full)
        neg_probs_comb = torch.cat(neg_probs_full)
        pos_scores_comb = torch.cat(pos_scores_full)
        neg_scores_comb = torch.cat(neg_scores_full)
        
        
        ### edges,edges_false only used for their length, so don't matter
        metrics = self.loss_handler(edges_comb,
                                    edges_false_comb,
                                    pos_probs_comb,
                                    neg_probs_comb,
                                    pos_scores_comb,
                                    neg_scores_comb,
                                    num_graphs = len(index_to_embeddings_dict)
                                    )
        if use_centroid_loss:
            metrics['loss'] = torch.Tensor([metrics['loss']])
            metrics['loss'].view((1, 1))
            link_prediction_loss = metrics['loss']
            graph_prediction_loss = centroid_loss
            total_loss = link_prediction_loss + graph_prediction_loss 
            graph_prediction_loss_fraction = graph_prediction_loss / total_loss
            link_prediction_loss_fraction = link_prediction_loss / total_loss
            print(f"Loss Fractions : LP : {link_prediction_loss_fraction}, GP : {graph_prediction_loss_fraction}")
            ALPHA_FOR_LOSS = 0.8
            weighted_loss = ALPHA_FOR_LOSS * link_prediction_loss + (1 - ALPHA_FOR_LOSS) * graph_prediction_loss
            metrics['loss'] = weighted_loss
            # TODO: Test how only GP without LP loss changes embeddings, currently seems to not learn properly.
            # It is possible that the autograd is getting messed up from recasting the torch tensor.
            # metrics['loss'] = centroid_loss
        return metrics
    
    def get_distance_between_embeddings(self, embeddings):
        dist = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist += self.manifold.sqdist(embeddings[i], embeddings[j], k = 1)
        
        return dist
    def get_edges(self, embeddings, data, split):
        if not self.is_inductive:
            edges_false = data[f'{split}_edges_false']
            edges = data[f'{split}_edges']
            
        else: ## because train / val splits naturally are unbalanced-- maybe try one without balancing?
            edges = data['edges'] 
            num_pos_edges = len(edges)
            num_neg_edges = len(data['edges_false'])
            # TODO: With MSE Loss (not Margin Loss), test whether better results without sampling/balance
            if self.args.use_margin_loss:
                edges_false = data['edges_false'][np.random.randint(0, num_neg_edges, num_pos_edges)] # Balancing Positive / Negative Edges
            else:
                edges_false = data['edges_false']
            # edges_false = data['edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)] # Balancing Positive / Negative Edges
            
        # sample_rate = 10
        # sample_rate = -1 ## for no sampling
        # if split == ('train') and (sample_rate > 0): 
        #     try:
        #         self.train_only
        #     except:
        #         self.train_only=False
        #     if self.train_only and not self.is_inductive:  
        #         splits = ['val','test']
        #         for s in splits:
        #             edges_false = torch.concat([edges_false,
        #                                         data[f'{s}_edges_false']])
                    
        #             edges = torch.concat([edges,data[f'{s}_edges']]) 
        #     edges_false = edges_false[np.random.randint(0, len(edges_false), len(edges)*sample_rate)]
            
        #     self.previous_edges_false = edges_false
        return edges, edges_false
    # Cole also has code for encoding node degree as features

    def loss_handler(self, 
                    edges,
                    edges_false,
                    pos_probs,
                    neg_probs,
                    pos_scores,
                    neg_scores,
                    num_graphs):
        # Cole mentioned using use_weighted loss
        self.args.use_weighted_loss = True
        if not self.args.use_margin_loss:
            
            if hasattr(self.args, 'use_weighted_loss') and self.args.use_weighted_loss:
                # logging.info("Now using PLV-derived labels and MSE loss for Fermi-Dirac Decoder probabilities")
                # Cole's way of computing loss function based on Fermi-Dirac Decoder
                # pos_probs extracted from min-maxing PLV matrices
                loss = F.mse_loss(pos_scores, pos_probs)
                neg_loss= F.mse_loss(neg_scores, neg_probs)
            else:
                # Chami's way of computing loss function based on Binary Cross Entropy Loss Function
                loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) 
                neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            loss += neg_loss
            if pos_scores.is_cuda:
                pos_scores = pos_scores.cpu()
                neg_scores = neg_scores.cpu()
        else:
            
            # NOTE: Does not use min-maxed probabilities from PLV Matrix, only binarized scores it seems 
            # Margin Loss requires predicted and true probs to be in same tensor.....
            # TODO: Make sure this is correct way of feeding into Margin Loss
            # loss = self.loss(torch.stack([pos_probs, pos_scores]))
            # neg_loss = self.loss(torch.stack([neg_probs, neg_scores]))
            
            preds = torch.stack([pos_scores, neg_scores], dim=-1)
            loss = self.loss(preds)
            # logging.info(f"Margin Loss: {loss}")
            
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = np.array(list(pos_scores.data.cpu().numpy()) + list(neg_scores.data.cpu().numpy()))
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        acc = self.binary_acc(preds, labels)

        # logging.info(f"Accuracy: {acc}") 
        
        
        # preds_binary = np.round(preds)
        
        # tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        # logging.info(f"Confusion Matrix :: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

        metrics = {'loss': loss, 
                    'roc': roc, 
                    'ap': ap,
                    'acc': acc,
                    'num_edges_true': len(edges),
                    'num_edges_false': len(edges_false),
                    'num_edges': len(edges) + len(edges_false),
                    'num_graphs': num_graphs
                    }
        
        return metrics
    
    def true_probs(self, adj_probs, idx):
        idx = idx.cpu().numpy()
        true_probs = adj_probs[idx[:,0],idx[:,1]]
        return true_probs
    
    def get_adj_prob(self, plv_matrix, is_stretch_sigmoid=False):
        # NOTE: If using identity matrix as PLV features, do not need to min_max_scale
        upper_triangular = []
        for i in range(plv_matrix.shape[0]):
            for j in range(i + 1, plv_matrix.shape[1]):
                upper_triangular.append(plv_matrix[i, j])
        data_flat = np.array(upper_triangular).flatten()
        
        def add_noise_func(plv_matrix, data_flat, rel_noise, mean = 0):
            dataset_std=(np.std(data_flat))
            noise_mat = np.random.normal(mean,
                                        dataset_std * rel_noise, 
                                        size = plv_matrix.shape)
            plv_matrix = plv_matrix + noise_mat
            return plv_matrix

        def mm_stretch(plv_matrix,
                    data_flat,
                    mm_min_pct = 5,
                    mm_max_pct = 95,
                    mm_clip_data = True,
                    mm_clip_range = (0,1),
                    identity_treatment='raw',
                    invert= False
                    ):  
            
            ## if you change data_flat to a single plv_matrix flat, you can retrofit to single norms
            plv_matrix_copy = deepcopy(plv_matrix)
            if identity_treatment == 'raw':
                pass
            elif identity_treatment == 'set_to_max':

                for s in range(plv_matrix_copy.shape[0]):
                    plv_matrix_copy[s,s] = np.max(data_flat)

            data_flat_use = 1 / data_flat if invert else data_flat
            plv_matrix_use = 1 / plv_matrix_copy if invert else plv_matrix_copy

            adj_prob = (plv_matrix_use - np.percentile(data_flat_use,mm_min_pct))/((np.percentile(data_flat_use,mm_max_pct)-np.percentile(data_flat_use,mm_min_pct)))
            if mm_clip_data:
                adj_prob = np.clip(adj_prob,
                                    mm_clip_range[0],
                                    mm_clip_range[1])
            return adj_prob

        def ms_sigmoid(plv_matrix,r,t, identity_treatment = '1'):
            
            plv_matrix_copy = deepcopy(plv_matrix)
            
            return  1 / (1.0 + np.exp(-((plv_matrix_copy - r) / t)))
            # adj_prob=sig_func_sim(plv_matrix)

            # if identity_treatment == '1':
            #     for s in range(adj_prob.shape[0]):
            #         adj_prob[s, s] = 1
            # return adj_prob
            
        def ms_norm(plv_matrix,
                    data_flat,
                    mm_clip_data=False,
                    mm_clip_range=(0,1),
                    identity_treatment='raw',
                    percentile_to_set=95
                    ):
            plv_matrix_copy = deepcopy(plv_matrix)
            if percentile_to_set < 1:
                percentile_to_set *= 100  ## should be full numbers!
            if identity_treatment == 'Raw':
                pass
            elif identity_treatment == 'set_to_pct':        
                for s in range(plv_matrix_copy.shape[0]):
                    plv_matrix_copy[s,s]=np.percentile(data_flat,percentile_to_set)
            adj_prob=(plv_matrix_copy-np.mean(data_flat))/(np.std(data_flat))
            if mm_clip_data:
                adj_prob=np.clip(adj_prob,mm_clip_range[0],mm_clip_range[1])
            return adj_prob

        # if add_noise > 0:
        #     ratio = -0.07
        #     mean_det = ratio * add_noise
        #     mean = np.random.uniform(mean_det, -mean_det / 6)
        #     single_scan = norm_functions['add_noise'](single_scan, data_flat, rel_noise=add_noise,mean=mean)
        #     adj_noise = single_scan

        # TODO: stretch_r and stretch_t should probably be fetched from train.py
        stretch_r = 2.0
        stretch_t = 1.0
        if is_stretch_sigmoid:
            adj_prob = ms_sigmoid(plv_matrix,
                            stretch_r,
                            stretch_t,
                            data_flat,
                            identity_treatment='1')
        else:
            STRETCH_LOSS_PERCENT = 95
            # logging.info(f"Min Max Stretching : {args.stretch_loss}")
            adj_prob= mm_stretch(plv_matrix,
                            data_flat,
                            mm_min_pct = 100 - STRETCH_LOSS_PERCENT,
                            mm_max_pct = STRETCH_LOSS_PERCENT,
                            mm_clip_data = True,
                            identity_treatment = 'raw') 
        ## seems like set to max is already covered
        return adj_prob    
    
    def report_epoch_stats(self):
        statistics = [self.epoch_stats['num_correct'], self.epoch_stats['num_total'], self.epoch_stats['loss']] ## could put back with train in (or not distributed)
        
        precision = float(self.epoch_stats['ap']) / self.epoch_stats['num_updates']
        roc = float(self.epoch_stats['roc']) / self.epoch_stats['num_updates']
        loss = float(self.epoch_stats['loss']) / self.epoch_stats['num_updates']
        acc =  float(self.epoch_stats['acc']) / float(self.epoch_stats['num_updates'])
        avg_stats = {
            'prefix': self.epoch_stats['prefix'],
            'epoch':  self.epoch_stats['epoch'],
            'loss': loss,
            'roc': roc,
            'ap': precision,
            'acc': acc,
        }
        # Fixed accuracy print issue, was not 0, it is the stat string formatting that was zeroing it out since it is a float between 0 and 1
        stat_string = "%s Phase of Epoch %d: Precision %.6f, ROC %.6f, Loss %.6f, Accuracy %.6f, Edges %d, Graphs %d" % (
                self.epoch_stats['prefix'],
                self.epoch_stats['epoch'],
                precision, 
                roc,
                loss,
                acc, 
                self.epoch_stats['num_total'],
                self.epoch_stats['num_graphs'])
        
        
        return avg_stats, stat_string ## still shows higher numbers but thats okay

    def update_epoch_stats(self, metrics, split):
        with torch.no_grad():
            ## if loss is mean but num total scales with batch size, will lead to problem

            old_total  = self.epoch_stats['num_total']
            new_total = old_total + metrics['num_edges']

            self.epoch_stats['ap'] += float(metrics['ap']) 

            self.epoch_stats['acc'] += metrics['acc']
            self.epoch_stats['roc'] += float(metrics['roc']) ### check this- is it average roc? then we need to weight per graph-- absolutely need to weight this- because we are probably skewed towards smaller graphs that have more edges

            self.epoch_stats['loss'] += float(metrics['loss'])
            
            self.epoch_stats['num_total'] = new_total
            self.epoch_stats['num_true'] += metrics['num_edges_true']
            self.epoch_stats['num_false'] += metrics['num_edges_false']
            self.epoch_stats['num_graphs'] += metrics['num_graphs']
            self.epoch_stats['num_updates'] += 1
        return self.epoch_stats
    
    def reset_epoch_stats(self, epoch, prefix):
        """
        prefix: train/val/test
        """
        ## why no keep track of all stats (roc, acc, prec) for all ?

        # if (prefix != 'start') and ('start' != self.epoch_stats['prefix']):
            # if self.epoch_stats['prefix'] not in self.metrics_tracker:
                # self.metrics_tracker[self.epoch_stats['prefix']] = []
        avg_stats, _ = self.report_epoch_stats()
        avg_stats['r'] = self.dc.r
        avg_stats['t'] = self.dc.t
            # self.metrics_tracker[self.epoch_stats['prefix']].append(avg_stats) ### adds new stats before resetting
            
        self.epoch_stats = {
            'prefix': prefix,
            'epoch': epoch,
            'loss': 0,
            'roc': 0,
            'ap': 0,
            'acc': 0,
            'num_correct': 0,## add to acc_f1 funct
            'num_true':0,
            'num_false':0,
            'num_graphs':0,
            'num_total': 0,
            'num_updates':0

        }
        return

    def compute_metrics_for_evaluation(self, embeddings, graph_data_dict, split):
        
        embeddings_list = [embeddings]
        graph_data_dicts = [graph_data_dict]

        metrics = self.compute_metrics_multiple(embeddings_list, 
                                                graph_data_dicts, 
                                                split)
        return metrics
    
    def evaluate_graph_data_dicts(self, epoch, graph_data_dicts, prefix, freeze=True):
        self.eval()
        setattr(self.args, 'currently_training', False)
        embeddings_list = []
        with torch.no_grad():
            self.reset_epoch_stats(epoch, prefix)
            for graph_data_dict in graph_data_dicts:

                embeddings = self.encode(
                    graph_data_dict['features'].to(self.args.device), 
                    graph_data_dict['adj_train_norm'].to(self.args.device)
                    )
                
                metrics = self.compute_metrics_for_evaluation(embeddings, graph_data_dict, prefix)
                self.update_epoch_stats(metrics, prefix)
                
                embeddings_list.append(embeddings)
            epoch_stats, stat_string = self.report_epoch_stats()
        return epoch_stats, embeddings, stat_string, embeddings_list

    def evaluate_graph_data_dicts_and_get_index_to_embeddings_dict(self, epoch, graph_data_dicts, prefix, freeze=True):
        self.eval()
        setattr(self.args, 'currently_training', False)
        embeddings_list = []
        index_to_embeddings_dict = dict()
        with torch.no_grad():
            self.reset_epoch_stats(epoch, prefix)
            for graph_data_dict in graph_data_dicts:

                embeddings = self.encode(
                    graph_data_dict['features'].to(self.args.device), 
                    graph_data_dict['adj_train_norm'].to(self.args.device)
                    )
                
                metrics = self.compute_metrics_for_evaluation(embeddings, graph_data_dict, prefix)
                self.update_epoch_stats(metrics, prefix)
                
                embeddings_list.append(embeddings)
                index_to_embeddings_dict[graph_data_dict['index']] = embeddings
            epoch_stats, stat_string = self.report_epoch_stats()
        return epoch_stats, embeddings_list, stat_string, index_to_embeddings_dict


    def binary_acc(self, predicted_labels, labels):
        num_correct = 0
        
        for i in range(len(labels)):
            predicted_label = predicted_labels[i]
            label = labels[i]
        
            if type(label)==float:                
                if len(label) > 1:
                    raise Exception('why are we getting a list')
                label = label[0]

            if round(predicted_label) == round(label): # So Cole is using a threshold of 0.5 for pos_scores/neg_scores
                num_correct += 1
        
        acc = float(num_correct) / float(len(labels))
        return acc
    
    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])


from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

from typing import List
import numpy as np
import torch
from tqdm import tqdm
from config import parser
from manifolds.base import ManifoldParameter
from models.base_models import NCModel, LPModel
from optim.radam import RiemannianAdam
from optim.rsgd import RiemannianSGD
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, get_dir_name_for_hyperparameter_search, get_dir_name_for_age_prediction
# FHNN does not use Frechet Mean calculation since it is still slow
# from diff_frech_mean.frechet_agg import frechet_B
import copy

import torch.nn.functional as F
from utils.eval_utils import acc_f1

DATAPATH = os.path.join(os.getcwd(), "data")
LOG_DIR = os.path.join(os.getcwd(), "logs")
BATCH_SIZE = 64
os.environ['DATAPATH'] = DATAPATH
os.environ['LOG_DIR'] = LOG_DIR

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
            # save_dir = get_dir_name_for_hyperparameter_search(models_dir, args)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Construct the path to the threshold file (threshold is used to convert PLV matrices to binary adjacency matrices)
    threshold_file = os.path.join(os.getcwd(), 'data', 'vci', 'vci_threshold.txt')

    # Load the threshold value from the file
    with open(threshold_file, 'r') as f:
        line = f.readline().strip()
        try:
            # If the line contains just a number (minimal format)
            threshold_value = float(line)
        except ValueError:
            # If the line has a label (descriptive format)
            threshold_value = float(line.split(':')[-1].strip())

    # Assign to args.threshold
    args.threshold = threshold_value

    # Load data
    args.use_super_node = False
    args.use_thickness = False
    args.use_myelins = False
    # args.use_margin_loss = True
    args.use_margin_loss = True
    args.use_batch_learning = True

    logging.info("Use Super Node : {}".format(args.use_super_node))
    logging.info("Use Batch Learning : {}".format(args.use_batch_learning))
    logging.info("Use Margin Loss : {}".format(args.use_margin_loss))
    logging.info("Use Cortical Thickness: {}".format(args.use_thickness))
    logging.info("Use Myelination : {}".format(args.use_myelins))
    logging.info("Step Size for Reduction Factor (Gamma) for learning rate : {}".format(args.lr_reduce_freq))
    logging.info("Reduction Factor (Gamma) for learning rate : {}".format(args.gamma))
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    
    train_graph_data_dicts, val_graph_data_dicts, test_graph_data_dicts = \
        data["train_graph_data_dicts"], data["val_graph_data_dicts"], data["test_graph_data_dicts"]

    train_feat_0 = train_graph_data_dicts[0]['features']

    args.n_nodes, args.feat_dim = train_feat_0.shape
    if args.task == 'nc':
        args.n_nodes, args.feat_dim = data['features'].shape
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(train_graph_data_dicts[0]['train_edges_false'])
        args.nb_edges = len(train_graph_data_dicts[0]['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs  #'reduce lr every lr-reduce-freq or None to keep lr constant'

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))

    def move_graph_data_dicts_to_device(graph_data_dicts):
        for graph_data in graph_data_dicts:
            for key in graph_data:
                if torch.is_tensor(graph_data[key]):
                    graph_data[key] = graph_data[key].to(args.device)
        return graph_data_dicts
    if args.cuda is not None and int(args.cuda) >= 0 :
        model = model.to(args.device)
        train_graph_data_dicts = move_graph_data_dicts_to_device(train_graph_data_dicts)
        val_graph_data_dicts = move_graph_data_dicts_to_device(val_graph_data_dicts)
        test_graph_data_dicts = move_graph_data_dicts_to_device(test_graph_data_dicts)
    
    num_train_graphs = len(train_graph_data_dicts)
    num_val_graphs = len(val_graph_data_dicts)
    num_test_graphs = len(test_graph_data_dicts)
    
    logging.info(f"Number of Train Graph Data Dicts in List: {num_train_graphs}")
    logging.info(f"Number of Validation Graph Data Dicts in List: {num_val_graphs}")
    logging.info(f"Number of Test Graph Data Dicts in List: {num_test_graphs}")
    
    train_indices = [train_graph_data_dict['index'] for train_graph_data_dict in train_graph_data_dicts]
    val_indices = [val_graph_data_dict['index'] for val_graph_data_dict in val_graph_data_dicts]
    test_indices = [test_graph_data_dict['index'] for test_graph_data_dict in test_graph_data_dicts]
    
    logging.info(f"Train Subject Indices {train_indices}")
    logging.info(f"Validation Subject Indices {val_indices}")
    logging.info(f"Test Subject Indices {test_indices}")

    # No Fermi-Dirac optimization for now since Cole eventually decided against it
    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        0.0
    }]
    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=args.lr,
                                   stabilize=10)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters,
                                  lr=args.lr,
                                  stabilize=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    
    args_to_save = copy.deepcopy(args)
    if hasattr(args_to_save,'pruner'): setattr(args_to_save,'pruner','Cannot serialize pruner')
    if hasattr(args_to_save,'sampler'): setattr(args_to_save,'sampler','Cannot serialize sampler')
    if hasattr(args_to_save,'trial'): setattr(args_to_save,'trial','Cannot serialize trial')
    if hasattr(args_to_save,'change_threshold'): setattr(args_to_save,'change_threshold','Cannot serialize function')

    args.frech_B_dict = dict()
    
    logging.info(f"Max Number of Epochs : {args.epochs}")
    for epoch in range(args.epochs): 
        # logging.info(f"Model Curvature: {model.c[0]}")
        model.train()
    
        t = time.time()
        
        if args.use_batch_learning:
            # NOTE: Use embeddings dictionary index -> embeddings instead
            # TODO: Check how embeddings dict does relative to list
            # train_embeddings_list = batch_learning(model, train_graph_data_dicts, optimizer, lr_scheduler, args)
            logging.info("Using batch learning index to embeddings dict for training")
            train_index_to_embeddings_dict = batch_learning_for_index_to_embeddings_dict(model, train_graph_data_dicts, optimizer, lr_scheduler, args)
        else:
            # train_embeddings_list = []
            train_index_to_embeddings_dict = dict()
            
            for train_graph_data in train_graph_data_dicts:
                # TODO: Figure out if zero gradient should happen here or at the beginning of the for train_graph_data for loop
                optimizer.zero_grad() 

                # Frechet Mean Aggregation Layer Memoizations:
                # Memoization to speed up training
                # if train_graph_data['index'] in model.args.frech_B_dict:
                    
                #     model.args.frechet_B = model.args.frech_B_dict[train_graph_data['index']]
                # # elif model.args.use_virtual:
                # #     model.args.frechet_B=361
                # else:
                #     model.args.frechet_B = frechet_B(train_graph_data['adj_train_norm'])
                    
                #     model.args.frech_B_dict[train_graph_data['index']] = model.args.frechet_B

                embeddings = model.encode(
                    train_graph_data['features'].to(args.device), 
                    train_graph_data['adj_train_norm'].to(args.device),
                    )
                train_index_to_embeddings_dict[train_graph_data['index']] = embeddings
                # train_embeddings_list.append(embeddings)
            # TODO: Make sure loss and gradient descent is done 
            # properly, so that the gradient is updated after running
            # all graphs in the batch
            # train_metrics = model.compute_metrics_multiple(
            #     train_embeddings_list,
            #     train_graph_data_dicts,
            #     'train'
            #     )
            train_metrics = model.compute_metrics_multiple_for_embeddings_dict(
                train_index_to_embeddings_dict,
                train_graph_data_dicts,
                'train'
                )
            train_metrics['loss'].backward()

            if model.c < 0.1:
                logging.info(f'Low Curvature: {model.c}')
            if model.c < 0.02:
                logging.info(f'Critically Low Curvature: {model.c}')
                # Clips the curvature back to at least 0.02, a bit suspicious.
                model.c = torch.clip(model.c, 0.02)

            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
            
            model.update_epoch_stats(train_metrics, 'train')         
        with torch.no_grad():
            if (epoch + 1) % args.log_freq == 0:        
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            val_embeddings_list = []
            val_index_to_embeddings_dict = dict()
            test_embeddings_list = []
            test_index_to_embeddings_dict = dict()
            # Evaluation Epoch :
            # Consider making evaluation epoch take place less frequently
            # args.eval_freq = 10 
            if (epoch + 1) % args.eval_freq == 0:
                epoch_time = int(time.time() - t)
                logging.info(f'Training Epoch Time : {epoch_time} seconds')
                
                # val_metrics, val_embeddings, val_string, val_embeddings_list = model.evaluate_graph_data_dicts(epoch, 
                #                                                 val_graph_data_dicts, 
                #                                                 'val', 
                #                                                 freeze=True)
                # test_metrics, test_embeddings, test_string, test_embeddings_list = model.evaluate_graph_data_dicts(epoch, 
                #                                                     test_graph_data_dicts, 
                #                                                     'test', 
                #                                                     freeze=False)
                
                val_metrics, val_embeddings, val_string, val_index_to_embeddings_dict = model.evaluate_graph_data_dicts_and_get_index_to_embeddings_dict(
                                                                epoch, 
                                                                val_graph_data_dicts, 
                                                                'val', 
                                                                freeze=True)
                test_metrics, test_embeddings, test_string, test_index_to_embeddings_dict = model.evaluate_graph_data_dicts_and_get_index_to_embeddings_dict(
                                                                epoch, 
                                                                test_graph_data_dicts, 
                                                                'test', 
                                                                freeze=False)
                logging.info(" ".join(['Val','Epoch: {:04d}'.format(epoch + 1), val_string]))
                logging.info(" ".join(['Test','Epoch: {:04d}'.format(epoch + 1), test_string]))
                if model.has_improved(best_val_metrics, val_metrics):
                    best_test_metrics = test_metrics
                    best_test_string = test_string
                    best_emb = [val_emb.cpu() for val_emb in val_embeddings]
                    if args.save:
                        # TODO: Check if avoiding constant saving saves time!
                        # np.save(os.path.join(save_dir, 'eval_embeddings.npy'), best_emb.detach().cpu().numpy())
                        pass
                    best_val_metrics = val_metrics
                    best_val_string = val_string
                    counter = 0
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break
        lr_scheduler.step()
        # torch.cuda.empty_cache()
    
    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # if not best_test_metrics:
    #     model.eval()
    #     best_emb = model.encode(data['features'], data['adj_train_norm'])
    #     best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    # logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    # logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    logging.info(" ".join(["Val set results:", best_val_string]))
    logging.info(" ".join(["Test set results:", best_test_string]))
    

    if args.save:
        np.save(os.path.join(save_dir, 'best_embeddings.npy'), best_emb)
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args_to_save), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")

        save_embeddings_dict(train_index_to_embeddings_dict, save_dir, 'train')
        save_embeddings_dict(val_index_to_embeddings_dict, save_dir, 'val')
        save_embeddings_dict(test_index_to_embeddings_dict, save_dir, 'test')
        
        # save_embeddings_list(train_embeddings_list, train_graph_data_dicts, save_dir, 'train')
        # save_embeddings_list(val_embeddings_list, val_graph_data_dicts, save_dir, 'val')
        # save_embeddings_list(test_embeddings_list, test_graph_data_dicts, save_dir, 'test')
        
        # TODO: See if need to save train, val, test graph data dicts or if they are not needed
        # pickle.dump(train_graph_data_dicts, open(os.path.join(save_dir, 'train_graph_data_dicts.pkl'), 'wb'))
        # pickle.dump(val_graph_data_dicts, open(os.path.join(save_dir, 'val_graph_data_dicts.pkl'), 'wb'))
        # pickle.dump(test_graph_data_dicts, open(os.path.join(save_dir, 'test_graph_data_dicts.pkl'), 'wb'))

        
    return best_val_metrics['loss']

def save_embeddings_dict(index_to_embeddings_dict, save_dir : str, split_str : str, ):
    if split_str not in ['train', 'val', 'test']: 
        raise AssertionError(f"Invalid split_str : {split_str} !")
    embeddings_folder_dir = os.path.join(save_dir, 'embeddings')
    os.makedirs(embeddings_folder_dir, exist_ok=True)
    logging.info(f"Saving {len(index_to_embeddings_dict)} {split_str} embeddings to {embeddings_folder_dir}")
    for embedding_index in index_to_embeddings_dict:
        embedding = index_to_embeddings_dict[embedding_index]
        np.save(os.path.join(embeddings_folder_dir, f'embeddings_{split_str}_{embedding_index}.npy'), embedding.cpu().detach().numpy())


def save_embeddings_list(embeddings_list, graph_data_dicts : List[dict], save_dir : str, split_str : str, ):
    if split_str not in ['train', 'val', 'test']: 
        raise AssertionError(f"Invalid split_str : {split_str} !")
    embeddings_folder_dir = os.path.join(save_dir, 'embeddings')
    os.makedirs(embeddings_folder_dir, exist_ok=True)
    logging.info(f"Saving {len(embeddings_list)} {split_str} embeddings to {embeddings_folder_dir}")
    for embedding_index, embedding in enumerate(embeddings_list):
        split_index = graph_data_dicts[embedding_index]['index']
        np.save(os.path.join(embeddings_folder_dir, f'embeddings_{split_str}_{split_index}.npy'), embedding.cpu().detach().numpy())

def get_embeddings_list_for_evaluation(model, 
                                    optimizer, 
                                    graph_data_dicts : List[dict]):
    embeddings_list = []
    for graph_data in graph_data_dicts:
        optimizer.zero_grad()

        # for key in graph_data:
        #     if torch.is_tensor(graph_data[key]):
        #         graph_data[key] = graph_data[key].to(args.device)

        embeddings = model.encode(
            graph_data['features'].to(args.device), 
            graph_data['adj_train_norm'].to(args.device)
            )
        
        embeddings_list.append(embeddings)
    return embeddings_list

def get_index_to_embeddings_dict_for_evaluation(model, 
                                    optimizer, 
                                    graph_data_dicts : List[dict]):
    embeddings_list = []
    index_to_embeddings_dict = dict()
    for graph_data in graph_data_dicts:
        optimizer.zero_grad()

        # for key in graph_data:
        #     if torch.is_tensor(graph_data[key]):
        #         graph_data[key] = graph_data[key].to(args.device)

        embeddings = model.encode(
            graph_data['features'].to(args.device), 
            graph_data['adj_train_norm'].to(args.device)
            )
        
        index_to_embeddings_dict[graph_data['index']] = embeddings
        embeddings_list.append(embeddings)
    return index_to_embeddings_dict

def run_data_single(data, index, model, skip_embedding=False, no_grad=False):
    
    data_i={}
    for k, vals in data.items():
        data_i[k] = vals[index]
        if torch.is_tensor(data_i[k]):
            data_i[k] = data_i[k].to(model.args.device)
    
    data_i['adj_mat'] = data_i['adj_mat'].to_sparse()
    data_i['adj_train_norm'] = data_i['adj_mat']

    # if data_i['graph_id'] in model.args.frech_B_dict:
    
    #     model.args.frechet_B = model.args.frech_B_dict[data_i['graph_id']]
    # elif model.args.use_virtual:
    
    #     model.args.frechet_B = 361
    # else:
    #     model.args.frechet_B = frechet_B(data_i['adj_train_norm'])
        
    #     model.args.frech_B_dict[data_i['graph_id']]=model.args.frechet_B
    
    edges_false_dict ={'train':{}}
    split='train'
    data_i['false_dict']= edges_false_dict[split]
    data_i['false_dict']= edges_false_dict['train']
    if skip_embedding:
        return None, data_i

    if no_grad:
        with torch.no_grad():
            embeddings = model.encode(data_i['features'].to(model.args.device), data_i['adj_mat'].to(model.args.device))
    else:
        embeddings = model.encode(data_i['features'].to(model.args.device), data_i['adj_mat'].to(model.args.device))
    
    return embeddings, data_i

def batch_learning(model, train_graph_data_dicts, optimizer, lr_scheduler, args):
    # batch_size = 64
    # num_epochs = 500
    # learning_rate = 0.001
    
    train_embeddings_list = []
    
    # torch.randperm(num_samples)
    # NOTE: No need to re-randomize, will only lose track of the indices since we already have the randomized train_graph_data_dicts 
    
    for i in range(0, len(train_graph_data_dicts), BATCH_SIZE):
        mini_batch_data = train_graph_data_dicts[i : i + BATCH_SIZE]
        batch_embeddings_list = []
        for batch_graph_data in mini_batch_data:
            embeddings = model.encode(batch_graph_data['features'].to(args.device),
                                    batch_graph_data['adj_train_norm'].to(args.device)
                        )
            batch_embeddings_list.append(embeddings)
            train_embeddings_list.append(embeddings)
        
        # TODO: Make sure loss and gradient descent is done 
        # properly, so that the gradient is updated after running
        # all graphs in the batch
        train_metrics = model.compute_metrics_multiple(
            batch_embeddings_list,
            mini_batch_data,
            'train'
            )
        
        
        train_metrics['loss'].backward()
        # Difference between above, and model.backward(train_metrics['loss'])
        # model.accumulate_gradients()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        model.update_epoch_stats(train_metrics, 'train')

    return train_embeddings_list
def batch_learning_for_index_to_embeddings_dict(model, train_graph_data_dicts, optimizer, lr_scheduler, args):
    # batch_size = 64
    # num_epochs = 500
    # learning_rate = 0.001
    
    train_embeddings_list = []
    train_index_to_embeddings_dict = dict()
    # torch.randperm(num_samples)
    # NOTE: No need to re-randomize, will only lose track of the indices since we already have the randomized train_graph_data_dicts 
    
    for i in range(0, len(train_graph_data_dicts), BATCH_SIZE):
        mini_batch_data = train_graph_data_dicts[i : i + BATCH_SIZE]
        # batch_embeddings_list = []
        batch_index_to_embeddings_dict = dict()
        for batch_graph_data in mini_batch_data:
            embeddings = model.encode(batch_graph_data['features'].to(args.device),
                                    batch_graph_data['adj_train_norm'].to(args.device)
                        )
            # batch_embeddings_list.append(embeddings)
            # train_embeddings_list.append(embeddings)
            train_index = batch_graph_data['index']
            train_index_to_embeddings_dict[train_index] = embeddings
            batch_index_to_embeddings_dict[train_index] = embeddings
        
        train_metrics = model.compute_metrics_multiple_for_embeddings_dict(
            batch_index_to_embeddings_dict,
            mini_batch_data,
            'train'
            )
        train_metrics['loss'].backward()
        # Difference between above, and model.backward(train_metrics['loss'])
        # model.accumulate_gradients()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        model.update_epoch_stats(train_metrics, 'train')
    
    return train_index_to_embeddings_dict
    
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    logging.info("Training Complete!")



if False:
    logging.info("Starting Age Prediction Regression")
    from age_predictor import Age_Predictor
    dt = datetime.datetime.now()
    date = f"{dt.year}_{dt.month}_{dt.day}"
    models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
    save_dir = get_dir_name_for_age_prediction(models_dir)
    log_num = save_dir.split("\\")[-1]
    linear_age_predictor = Age_Predictor(date, log_num, "linear", "HR", "Cam-CAN")
    ridge_age_predictor = Age_Predictor(date, log_num, "ridge", "HR", "Cam-CAN")
    linear_mae, linear_mse, linear_correlation, linear_r2 = linear_age_predictor.regression()
    ridge_mae, ridge_mse, ridge_correlation, ridge_r2  = ridge_age_predictor.regression()
    logging.info(f"Linear Model Age Prediction MAE : {linear_mae}")
    logging.info(f"Linear Model Age Prediction MSE : {linear_mse}")
    logging.info(f"Linear Model Age Prediction Correlation : {linear_correlation}")
    logging.info(f"Linear Model Age Prediction R^2 Value : {linear_r2}")
    logging.info(f"Ridge Model Age Prediction MAE : {ridge_mae}")
    logging.info(f"Ridge Model Age Prediction MSE : {ridge_mse}")
    logging.info(f"Ridge Model Age Prediction Correlation : {ridge_correlation}")
    logging.info(f"Ridge Model Age Prediction R^2 Value : {ridge_r2}")

# Useful functions for working with graphs

import torch
import torch_geometric
from torch_geometric.data import Data,Dataset
from torch_geometric.utils import from_networkx, to_networkx

from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Planetoid

import networkx as nx

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from utils import file_name_generator
from utils import get_best_acc_from_models

import os
import time
import random
import json
import pickle
import copy
import os.path as osp
from pprint import pprint

# TODO: best practice would be a class from pytorch data?, write the methods to it
# how would we handle the nx part? could be stored within the class?

def node2vec_hyper_paramter_tuning():
    '''
    maybe optina tuning for the node2vec model? TODO
    we need to get the model, the data, the parameters to try
    when training, it should log and save the parameeters and the results 
    it should be stoppable and resumable
    the model should be able to be saved and loaded
    '''

    pass

def save_model(model: torch.nn.Module, filename: str) -> None:
    """
    Saves the model to the specified path.
    
    Parameters:
        model (torch.nn.Module): The PyTorch model to be saved.
        filename (str): The path where the model will be saved.
    Returns:
        None
    """
    
    # maybe a threshold for the evalution of the model? if its better than the latest one then save it TODO

    if not filename:
        filename = f"model_{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    else:
        filename = f"model_{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), filename)

def load_dataset(dataset_str = 'Cora'): 
    dataset_str = 'Cora'
    path = osp.join('.', 'training_data', 'datasets', dataset_str)
    dataset = Planetoid(path, dataset_str)
    data = dataset[0]
    data.creation_function = "load_dataset_" + dataset_str
    #dataset_name_str = str(getattr(data, 'creation_function', 'Unknown'))
    return dataset,data

def set_loader_and_optimizer(model):
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0) 
    # num_workers=0 is necessary for Windows
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    return loader, optimizer

def  model_init(model_init_params, data):
    """
    Input is the model json file. This function creates the model_init_params dictionary
    Afther that it creates the string for the filename to import the model and returns the initilaized model

    """
    # Load the model initialization parameters from the JSON file

    if isinstance(model_init_params, str):
        try:
            with open(model_init_params, 'r') as f:
                model_init_params = json.load(f)
            print("Model initialization parameters loaded:")
            print(model_init_params)
        except FileNotFoundError:
            print(f"Error: Model initialization parameters file not found at {model_init_params}")
    
        model = Node2Vec(data.edge_index, **model_init_params)
        print("Model initialized")

        model_state_dict_path = model_init_params.replace('.json', '.pth')

        try:
            model.load_state_dict(torch.load(model_state_dict_path))# map_location=device))
            print(f"Model state loaded from file: {model_state_dict_path}")
        except FileNotFoundError:
            print(f"Error: Model state file (.pth) not found at {model_state_dict_path}")
            return None # Or raise error
        except Exception as e:
            print(f"Error loading model state_dict: {e}")
            return None # Or raise error
        

        model.eval() # Set model to evaluation mode after loading state
        return model, model_init_params
    
    elif isinstance(model_init_params, dict):
        print("Model initialization parameters provided as dictionary.")

        model = Node2Vec(data.edge_index, **model_init_params)
        print("Model initialized")
        model.eval() # Set model to evaluation mode after loading state
        return model


def train_n2v(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader): 
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test_n2v(model, data):
    model.eval()
    z = model()
    acc = model.test(z[data.train_mask], data.y[data.train_mask],
                     z[data.test_mask], data.y[data.test_mask],
                     max_iter=150)
    return acc

def model_training_n2v(model, model_init_params, data, loader, optimizer, num_epochs, device, all_plots:bool = False, model_save_path='node2vec_best_model_00'):
    """
    Trains the Node2Vec model, saves the best state, and plots metrics.

    Args:
        model: The Node2Vec model instance.
        loader: The data loader for training.
        optimizer: The optimizer for the model.
        test_func: The function to evaluate the model (e.g., the test() function).
        num_epochs (int): The number of epochs to train for.
        device: The device ('cuda' or 'cpu') to train on.
        model_save_path (str): Path to save the best model state dictionary.

    Returns:
        tuple: A tuple containing:
            - dict: The state dictionary of the best performing model.
            - dict: A dictionary containing lists of losses, accuracies, and epoch times.
    """
    # TODO: make plotting optional with a flag
    
    start_time = time.time()
    losses = []
    accuracies = []
    epoch_times = []
    best_acc = 0.0
    best_model_state = None

    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        best_epoch = 1

        epoch_start_time = time.time()

        # --- Training Step --- TODO: could be a function like test
        model.train()
        total_loss = 0
        # Assuming tqdm is imported if you want the progress bar
        for pos_rw, neg_rw in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs} Training", leave=False):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        # --- End Training Step ---

        # --- Testing Step ---
        acc = test_n2v(model, data) # Call the provided test function
        accuracies.append(acc)
        # --- End Testing Step ---

        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)

        print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Duration: {epoch_duration:.2f}s')
        
        # Save the best model
        # best_acc_from_models = get_best_acc_from_models()

        # TODO: how to do the the checking to know if we need to save? create function to get it with same other data
        # also generalize the p q comparison notebook, run it overnight?
        # check name string of same params model to see if this is better
        
        if acc > best_acc:
            best_acc_str = f"_acc_{acc:.4f}"
            dataset_name_str = str(getattr(data, 'creation_function', 'Unknown'))
            print(f"Dataset name: {dataset_name_str}")
            model_name = "node2vec"

            best_acc = acc
            
            best_model_state = copy.deepcopy(model.state_dict())
            #  model_save_path='./training_data/models/node2vec_'
            with open(model_save_path + file_name_generator(model_init_params, model_name, dataset_name_str,best_acc_str) + '.json', "w") as f:
                json.dump(model_init_params, f, indent=4)
            
            # Save the model state dictionary
            torch.save(best_model_state, model_save_path + file_name_generator(model_init_params, model_name, dataset_name_str,best_acc_str) + '.pth')
            
            best_epoch = epoch
            print(f'    New best model saved with accuracy: {best_acc:.4f}')

    elapsed_time = time.time() - start_time
    print(f'\nTraining finished.')
    print(f'Total elapsed time: {elapsed_time:.2f} seconds')
    print(f'Best test accuracy: {best_acc:.4f}')
    print(f"Best model state saved to '{model_save_path}'")

    # --- Plotting ---
    # Get p and q values for color logic
    p_val = model_init_params.get('p', None)
    q_val = model_init_params.get('q', None)
    
    if (q_val is not None and p_val is not None and q_val > p_val):
        acc_color = 'b' 
    elif (p_val is not None and q_val is not None and p_val > q_val):
        acc_color = 'r'
    else:
        acc_color = 'g'
        

    if all_plots:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot Loss
        axs[0].plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', label='Loss')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Training Loss per Epoch')
        axs[0].grid(True)
        axs[0].legend()

        # Plot Accuracy
        axs[1].plot(range(1, num_epochs + 1), accuracies, marker='o', linestyle='-', color=acc_color, label='Accuracy')
        axs[1].set_ylabel('Test Accuracy')
        axs[1].set_title('Test Accuracy per Epoch')
        axs[1].grid(True)
        axs[1].legend()

        # Plot Time
        axs[2].plot(range(1, num_epochs + 1), epoch_times, marker='o', linestyle='-', color='g', label='Epoch Duration')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Time (seconds)')
        axs[2].set_title('Epoch Duration')
        axs[2].grid(True)
        axs[2].legend()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(range(1, num_epochs + 1), accuracies, marker='o', linestyle='-', color=acc_color, label='Accuracy')
        ax.set_ylabel('Test Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_title('Test Accuracy per Epoch')
        ax.grid(True)
        
        # Add dataset name to the legend as a separate entry
        dataset_name = getattr(data, 'creation_function', 'Unknown')
        ax.plot([], [], ' ', label=f"Dataset: {dataset_name}")

        # Add p and q to the legend as a separate entry
        pq_text = f"p = {p_val}, q = {q_val}"
        ax.plot([], [], ' ', label=pq_text)  # Invisible plot for legend entry
        ax.legend()

    plt.tight_layout()
    plt.show()
    # --- End Plotting ---

    history = {
        'losses': losses,
        'accuracies': accuracies,
        'epoch_times': epoch_times
    }

    # Load the best state back into the model before returning (optional, but good practice)
    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_model_state, history

def create_parameters_dict(
    p_range=[1, 2],  # [0.5, 1, 2, 5]
    q_range=[1, 2],  # [start, stop, step] for q (default: 1, 101, 100)
    base_params=None,
    range_mode: bool = False
) -> dict:
    """
    Generates a dict of dicts of parameter sets for grid search over p and q.

    Args:
        p_range (list): If range_mode is True, [start, stop, step] for p values (inclusive of start, exclusive of stop).
                        If range_mode is False, a list of p values to use directly.
        q_range (list): If range_mode is True, [start, stop, step] for q values (inclusive of start, exclusive of stop).
                        If range_mode is False, a list of q values to use directly.
        base_params (dict): Base parameters to use for all sets (except p and q).
        range_mode (bool): If True, use range(start, stop, step). If False, treat p_range and q_range as value lists.

    Returns:
        dict: Dictionary where keys are 'p={p}_q={q}' and values are parameter dicts.
    """
    skip_equal = True  # Set to True if you want to skip equal p and q values

    if base_params is None:
        base_params = {
            "embedding_dim": 128,
            "walk_length": 70,
            "context_size": 14,
            "walks_per_node": 18,
            "num_negative_samples": 1,
            "sparse": True
        }

    if range_mode:
        p_values = list(range(p_range[0], p_range[1], p_range[2]))
        q_values = list(range(q_range[0], q_range[1], q_range[2]))
    else:
        p_values = list(p_range)
        q_values = list(q_range)

    print(f"p_values: {p_values}")
    print(f"q_values: {q_values}")

    parameter_dicts = {}
    for p in p_values:
        for q in q_values:
            if skip_equal:
                if p == q:
                    continue  # Skip if p and q are equal
            params = base_params.copy()
            params['p'] = p
            params['q'] = q
            key = f"p={p}_q={q}"
            parameter_dicts[key] = params

    return parameter_dicts

def create_parameters_dict_old() -> dict:
    """
    Converts a dictionary of parameters into a string format for easy saving.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        dict: Dictionary with string keys and values.
    """
    #  d = 128, r = 10, l = 80, k = 10
    # pq in {0.25, 0.50, 1, 2, 4}

    model_init_params_q = { # close, bfs
    "embedding_dim": 128,
    "walk_length": 20,
    "context_size": 10,
    "walks_per_node": 20,
    "num_negative_samples": 1,
    "p": 1,
    "q": 200,
    "sparse": True
    }

    model_init_params_p = { # exploration, dfs
        "embedding_dim": 128,
        "walk_length": 20,
        "context_size": 10,
        "walks_per_node": 20,
        "num_negative_samples": 1,
        "p": 200,
        "q": 1,
        "sparse": True
        }

    parameter_dicts = {}
    parameter_dicts['q'] = model_init_params_q
    parameter_dicts['p'] = model_init_params_p

    # params_str = {str(k): str(v) for k, v in params.items()}
    
    return parameter_dicts
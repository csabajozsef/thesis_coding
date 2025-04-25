# Useful functions for working with graphs

import torch
import torch_geometric
from torch_geometric.data import Data,Dataset
from torch_geometric.utils import from_networkx, to_networkx
from networkx.algorithms.community.modularity_max import greedy_modularity_communities

from torch_geometric.nn import Node2Vec
from torch_geometric.datasets import Planetoid

import networkx as nx

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from utils import params_to_string

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

def load_dataset_cora(dataset = 'Cora'): 
    dataset = 'Cora'
    path = osp.join('.', 'training_data', 'datasets', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]
    return data

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

def model_training_n2v(model, model_init_params, data, loader, optimizer, num_epochs, device, model_save_path='node2vec_best_model_00'):
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
        if acc > best_acc:
            best_acc = acc
            
            # Use deepcopy to ensure the state is saved correctly at this point
            best_model_state = copy.deepcopy(model.state_dict())
            # Save init parameters to json
            with open(model_save_path + params_to_string(model_init_params) + '.json', "w") as f:
                json.dump(model_init_params, f, indent=4)
            
            # Save the model state dictionary
            torch.save(best_model_state, model_save_path + params_to_string(model_init_params) + '.pth')
            
            best_epoch = epoch
            print(f'    New best model saved with accuracy: {best_acc:.4f}')

    elapsed_time = time.time() - start_time
    print(f'\nTraining finished.')
    print(f'Total elapsed time: {elapsed_time:.2f} seconds')
    print(f'Best test accuracy: {best_acc:.4f}')
    print(f"Best model state saved to '{model_save_path}'")

    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Loss
    axs[0].plot(range(1, num_epochs + 1), losses, marker='o', linestyle='-', label='Loss')
    axs[0].set_ylabel('Average Loss')
    axs[0].set_title('Training Loss per Epoch')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Accuracy
    axs[1].plot(range(1, num_epochs + 1), accuracies, marker='o', linestyle='-', color='r', label='Accuracy')
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

def add_greedy_modularity_labels_nx(G: nx.Graph) -> nx.Graph:
    """
    Adds node labels ('y' attribute) to a NetworkX graph based on
    communities found using the greedy modularity maximization algorithm.

    Args:
        G (nx.Graph): The input NetworkX graph.

    Returns:
        nx.Graph: The input graph with the 'y' node attribute added,
                  containing community labels for each node.
                  Returns the original graph if community detection fails.
    """
    if not isinstance(G, nx.Graph):
        print("Error: Input must be a NetworkX Graph object.")
        return G

    try:
        # Find communities using greedy modularity maximization
        # Ensure the graph is undirected for the algorithm if necessary
        # If your graph might be directed, consider converting: G_undirected = G.to_undirected()
        communities = greedy_modularity_communities(G) # Use G or G_undirected

        # Create a partition dictionary mapping node to community ID
        partition = {}
        for i, community_nodes in enumerate(communities):
            for node in community_nodes:
                partition[node] = i # Assign community index as label

        # Set the 'y' attribute on the NetworkX graph nodes
        nx.set_node_attributes(G, partition, 'y')
        print(f"Successfully added 'y' attribute to NetworkX graph with {len(communities)} communities found.")

    except Exception as e:
        print(f"An error occurred during label creation: {e}")
        # Optionally return original graph or raise error

    return G

def create_parameters_dict() -> dict:
    """
    Converts a dictionary of parameters into a string format for easy saving.

    Args:
        params (dict): Dictionary of parameters.

    Returns:
        dict: Dictionary with string keys and values.
    """

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
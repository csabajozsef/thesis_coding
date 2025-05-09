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

import os
import time
import random
import json
import pickle
import copy
import os.path as osp
from pprint import pprint

def set_seeds_and_device():
    """Set the random seed for reproducibility."""
    # Set the random seed for reproducibility
    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed
    # https://pytorch.org/docs/stable/cuda.html#torch.cuda.manual_seed_all
    # https://pytorch.org/docs/stable/generated/torch.cuda.html#torch.cuda.manual_seed_all
    # https://pytorch.org/docs/stable/generated/torch.random.html#torch.random.seed
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    print(f"Random seed set to: {SEED}")

    return device

def y_attribute_checker(graph:nx.Graph) -> bool:
    '''
    This function checks if the y attribute is present in the graph.
    '''
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a networkx graph object")
    
    # Check if 'y' attribute exists for all nodes
    for node in graph.nodes():
        if 'y' not in graph.nodes[node]:
            print(f"Node {node} does not have 'y' attribute.")
            return False
    return True

def nx_to_pytorch_data_converter(g:nx.Graph) -> Data:
    '''
    This function converts a networkx graph to a pytorch geometric graph.
    TODO: vice versa?
    '''
    assert isinstance(g, nx.Graph), "Graph must be a networkx graph object"
    # y should be the transferred node labels, if they exist

    # y = torch.tensor([g.nodes[node]['y'] for node in g.nodes], dtype=torch.long)
    # g.graph["num_classes"] = len(set(y.numpy()))
    
    # TODO: collect all node attributes and add them to the data object
    data = from_networkx(g, group_node_attrs = ['y']) 

    if y_attribute_checker(g) == True:
        # Check if 'y' attribute exists in the graph
        y = torch.tensor([g.nodes[node]['y'] for node in g.nodes], dtype=torch.long)
        data.y = y
    else:
        # If 'y' attribute is not found, raise an error or handle it accordingly
        raise AttributeError("The 'y' attribute is missing for some nodes in the graph.")
    
    if g.graph["creation_function"] is not None:
        data.creation_function = g.graph["creation_function"]

    return data

def create_masks(data, train_ratio=0.7):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

def params_to_string(params: dict, acc = None) -> str:
    """
    Converts a dictionary of parameters to a string in the format:
    "param1name_param1value_param2name_param2value..."
    """
    string_to_return = "_".join(f"{k}_{v}" for k, v in params.items())
    
    if acc is not None:
        string_to_return += "_"+str(acc)
        # add acc to string
    
    return string_to_return

def get_best_acc_from_models(params: dict):
    
    pass

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
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

def nx_to_pytorch_data_converter(g):
    '''
    This function converts a networkx graph to a pytorch geometric graph.
    TODO: vice versa?
    '''
    assert isinstance(g, nx.Graph), "Graph must be a networkx graph object"
    # y should be the transferred node labels, if they exist

    # y = torch.tensor([g.nodes[node]['y'] for node in g.nodes], dtype=torch.long)
    # g.graph["num_classes"] = len(set(y.numpy()))

    data = from_networkx(g, group_node_attrs = ['y'])

    if not hasattr(data, 'y') or data.y is None:
        print("Warning: 'y' attribute not found or is None after from_networkx. Attempting manual extraction.")
        try:
            y_values = [g.nodes[node]['y'] for node in g.nodes()]
            data.y = torch.tensor(y_values, dtype=torch.long)
        except KeyError:
            raise AttributeError("Failed to manually extract 'y' attribute. Check node data in NetworkX graph.")
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

def params_to_string(params: dict) -> str:
    """
    Converts a dictionary of parameters to a string in the format:
    "param1name_param1value_param2name_param2value..."
    """
    return "_".join(f"{k}_{v}" for k, v in params.items())

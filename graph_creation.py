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

def y_label_maker(graph:nx.Graph) -> nx.Graph:
    '''
    This function creates the y labels for the nodes in the graph.
    '''
    y = []
    for community_label, community in enumerate(graph.graph["partition"], start=1):
        y.extend([community_label] * len(community))

    nx.set_node_attributes(graph, {i: {'y': label} for i, label in enumerate(y)})

    return graph

def graph_with_communities_generator_sb(num_of_communities:int = 10, nodes_per_community:int = 10, probs = None, save_to_file:bool= False, print_text:bool = True) -> nx.Graph:
    '''
    This function generates a graph with communities using the stochastic block model.
    '''
    if probs is None:
        # Default: High intra-community probability, low inter-community probability
        if print_text:
            print(" => Using default probabilities in the stochastic block model")
        probs = np.full((num_of_communities, num_of_communities), 0.01)
        np.fill_diagonal(probs, 0.5)
    
    G = nx.stochastic_block_model(sizes=[nodes_per_community] * num_of_communities, p=probs)
    
    # Assign class labels to nodes based on communities (1, 2, 3...)
    y = []
    for community_label, community in enumerate(G.graph["partition"], start=1):
        y.extend([community_label] * len(community))

    nx.set_node_attributes(G, {i: {'y': label} for i, label in enumerate(y)})

    # G.graph["num_classes"] = num_of_communities
    G.graph["nodes_per_community"] = nodes_per_community
    G.graph["num_of_communities"] = num_of_communities
    # Before saving:
    G.graph["partition"] = [list(community) for community in G.graph["partition"]]

   
    #for key, value in G.graph.items():
    #   print(f"{key}: {value}")
    
    if save_to_file:
        # Save the graph to a file

        # os.makedirs("graphs", exist_ok=True)
        path = f"training_data/datasets/graphs/community_graph_{str(num_of_communities)}_{str(nodes_per_community)}.gpickle"

        with open(path, "wb") as f:
            pickle.dump(G, f)
        print(f" => Graph saved to {path}")
    return G   

def graph_with_hierarchy_generator(r:int = 3, h:int = 3, extra_edges:bool = True, print_text:bool = True) -> nx.Graph:
    '''
    This funtion generates a graph with a hierarchy.
    '''
    # Generate a large DFS-style graph with tree structure + structural links
    graph_dfs = nx.balanced_tree(r, h)  # ≈ 16,383 nodes ≈ r^h
    # r:Branching factor of the tree; each node will have r children.
    # h:Height of the tree
    # Add horizontal connections (between siblings) to slightly reduce diameter

        # Assign class labels based on depth-level (1, 2, 3...)
    y = {}
    if print_text:
        print(len(graph_dfs.nodes()))

    for node in graph_dfs.nodes():
        depth = nx.shortest_path_length(graph_dfs, source=0, target=node)
        y[node] = depth + 1  # Depth-level starts at 1

    nx.set_node_attributes(graph_dfs, {node: {'y': label} for node, label in y.items()})
    graph_dfs.graph["num_classes"] = h
    
    if extra_edges:
        extra_edges = []
        for i in range(0, graph_dfs.number_of_nodes(), 2):
            if i + 1 < graph_dfs.number_of_nodes():
                extra_edges.append((i, i+1))
        graph_dfs.add_edges_from(extra_edges)

    return graph_dfs

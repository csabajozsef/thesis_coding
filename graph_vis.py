
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

def print_graph_info_basic(data: Data|Dataset|nx.Graph)->None:
    """
    Prints information about a PyTorch Geometric Data object.
    
    Parameters:
        data (torch_geometric.data.data.Data): The PyTorch Geometric Data object containing graph information.
    Returns:
        None
    """

    # show_distribution parameter: we could show the node degreee distributions TODO
    # show_distribution parameter: we could show the ?  TODO

    # TODO: add a parameter to show the distribution of the node features and the edge features
    # TODO: change to print graph metrics, should worl on nx.Graph and torch_geometric.data.Data and contain a parameter for connectetness calc: such as global clustestin coefficien
    try:
        if isinstance(data, Data):
            print("Type: ",type(data))
            print(data)
            print("Keys of the data object:", data.keys())
            print("")
            print("Has isolated nodes:", data.has_isolated_nodes())
            print("Has self loops:", data.has_self_loops())
            print("Is directed:", data.is_directed())
            print("")
            print(f'Number of nodes: {data.num_nodes}')
            print(f'Number of edges: {data.num_edges}')
            print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
            print(f'Number of node features: {data.num_node_features}')
            print(f'Number of edge features: {data.num_edge_features}')
            print(f'Number of classes: {data.num_classes}')

        elif isinstance(data, Dataset):

            if len(data) == 1:

                print("Type: ",type(data))
                print("Number of graphs: ",len(data))
                print(data)
                
                print("Keys of the data object:", data[0].keys())
                print("")
                print("Has isolated nodes:", data[0].has_isolated_nodes())
                print("Has self loops:", data[0].has_self_loops())
                print("Is directed:", data[0].is_directed())
                print("")
                print(f'Number of nodes: {data[0].num_nodes}')
                print(f'Number of edges: {data[0].num_edges}')
                print(f'Average node degree: {data[0].num_edges / data[0].num_nodes:.2f}')
                print(f'Number of node features: {data[0].num_node_features}')
                print(f'Number of edge features: {data[0].num_edge_features}')
                print(f'Number of classes: {data.num_classes}')
            
            else:
                # multiple graphs in the dataset
                print("Multiple graphs in the dataset")
                print("--------------------")
                print("Type: ",type(data))
                print(data)
                print("Number of graphs: ",len(data))
                print(f"Number of classes in the dataset: {data.num_classes}")
                print(f"Number of node features in the dataset: {data.num_node_features}")
                # cumulatve stats on all graphs
                max_graph = max(data, key=lambda graph: graph.num_nodes)
                min_graph = min(data, key=lambda graph: graph.num_nodes)
                average_node_count = sum(graph.num_nodes for graph in data) / len(data)
                max_nodes = max_graph.num_nodes
                min_nodes = min_graph.num_nodes

                print(f'Most nodes in a graph: {max_nodes:.2f}')
                print(f'Least nodes in a graph: {min_nodes:.2f}')
                print(f'Average node count: {average_node_count:.2f}')

        elif isinstance(data, nx.Graph):
            print("Type: ",type(data))
            print(data)
            if data.graph.get("creation_function"):
                print("Graph creation function: ",data.graph["creation_function"])
            print("Number of nodes: ",data.number_of_nodes())
            print("Number of edges: ",data.number_of_edges())
            print("Average node degree: ",np.mean([d for n, d in data.degree()]))
            print("Has isolated nodes: ",len(list(nx.isolates(data))))
            print("Has self loops: ",len(list(nx.selfloop_edges(data))))
            print("Is directed: ",data.is_directed())
        else:
            print("Unhandled data type")
            print("Type: ",type(data))
    except AttributeError as e:
        print(f"AttributeError: {e}")

def print_graph_info_cluster(graph:nx.Graph|str, print_text:bool = False) -> None:
    '''
    This function performs graph analytics on the given graph. 
    '''
    print("[INFO] Starting graph analytics")
    if isinstance(graph, str):
        assert graph.endswith(".gpickle"), "Graph file must be in .gpickle format"
        with open(graph, "rb") as f:
            G = pickle.load(graph)
        if print_text:
            print(f"[INFO] Graph loaded from {graph}")
    elif isinstance(graph, Data):
        # G = nx.from_pyg_data(graph)
        print("[INFO] Graph is a PyTorch Geometric Data object, converting to NetworkX graph.")
        G = to_networkx(graph)
        if print_text:
            print(f"[INFO] Graph loaded from {graph}")
    elif isinstance(graph, nx.Graph):
        G = graph
        if print_text:
            print(f"[INFO] Graph loaded from {graph}")
    else:
        raise ValueError("Graph must be a networkx graph object or a file path to a .gpickle file")        

    print()
    print("----------Basic graph information-----------")
    print_graph_info_basic(G)
    # print()
    graph_stats = {}

    # Calculate the number of connected components
    try:
        if not G.is_directed():
            num_connected_components = nx.number_connected_components(G)
            graph_stats["Number of connected components"] = num_connected_components

            largest_cc = G.subgraph(max(nx.connected_components(G), key=len))
            graph_stats["Number of nodes in largest component"] = largest_cc.number_of_nodes()
        else:
            print("Warning: Connected components and largest component stats not available for directed graphs.")
            graph_stats["Number of connected components"] = "N/A (directed graph)"
            graph_stats["Number of nodes in largest component"] = "N/A (directed graph)"
    except Exception as e:
        print(f"Error calculating connected components: {e}")
        graph_stats["Number of connected components"] = "Error"
        graph_stats["Number of nodes in largest component"] = "Error"
    # Initialize dictionary for statistics and timing
    
    # Time each statistic calculation
    # graph_stats["Number of nodes"] = G.number_of_nodes()
    # graph_stats["Number of edges"] = G.number_of_edges()
    
    try:
        start_time = time.time()
        graph_stats["Average Clustering Coefficient"] = nx.average_clustering(G)
        clustering_time = time.time() - start_time
    except Exception as e:
        print(f"Error calculating Average Clustering Coefficient: {e}")
        clustering_time = None

    try:
        start_time = time.time()
        graph_stats["Transitivity/Global clustering coeff"] = nx.transitivity(G)
        transitivity_time = time.time() - start_time
    except Exception as e:
        print(f"Error calculating Transitivity/Global clustering coeff: {e}")
        transitivity_time = None

    try:
        start_time = time.time()
        graph_stats["Average Shortest Path (Largest Component)"] = nx.average_shortest_path_length(largest_cc)
        path_time = time.time() - start_time
    except Exception as e:
        print(f"Error calculating Average Shortest Path (Largest Component): {e}")
        path_time = None

    try:
        start_time = time.time()
        graph_stats["Number of Connected Components"] = nx.number_connected_components(G)
        cc_time = time.time() - start_time
    except Exception as e:
        print(f"Error calculating Number of Connected Components: {e}")
        cc_time = None
     # Store timing information
    
    bfs_stat_times = {}
    bfs_stat_times = {
        "Clustering calculation": clustering_time,
        "Path length calculation": path_time,
        "Components calculation": cc_time,
        "Transitivity calculation": transitivity_time
    }

    print("----------Graph extra statistics-----------")
    for k, v in graph_stats.items():
        print(f"{k}: {v}")
        # print(f"Time taken: {graph_stats[k]} seconds")
        # # plot?
        # print("\n")
    print()

def pyg_graph_data_visualizer(data: Data|Dataset)->None:
    """
    Visualizes the graph data using the PyTorch Geometric Data visualizer.
    
    Parameters:
        data (torch_geometric.data.data.Data): The PyTorch Geometric Data object containing graph information.
    Returns:
        None
    """
    # Visualize the graph data with networkx, 
    # inherit from networkx draw funtion and expand with automatic rewrite from pytorch form? TODO
    
    # node degree distribution
    if data.y is not None:
        print("Node type information available for histogram.")
        df = pd.DataFrame({'node_type': data.y.numpy()})
        # this could be just the built in data.num_classes or data.num_node_features - which in which case? TODO
        node_type_counts = df.groupby('node_type').size().reset_index(name='count')

        print("" ,node_type_counts)

        node_type_counts.plot(kind='bar', x='node_type', y='count', legend=False)

        plt.xlabel('Node Type')
        plt.ylabel('Count')
        plt.title('Histogram of Node Types in graph')
        plt.show()
    else:
        print("No node type information available for histogram.")
    # Calculate node degrees
    degrees = data.edge_index[0].bincount().numpy()

    # Create a histogram for node degree distribution
    plt.figure(figsize=(8, 6))
    plt.hist(degrees, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel('Node Degree')
    plt.ylabel('Frequency')
    plt.title('Node Degree Distribution')
    plt.show()

def graph_visualizer(G: nx.Graph, layout:str ="spring" "", save_to_file:bool= False) -> None:
    '''
    This function visualizes the graph with communities using NetworkX and Matplotlib.
    '''
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "bfs":
        pos = nx.bfs_layout(G,0)
    else:
        raise ValueError("Invalid layout type. Choose from 'spring', 'circular', 'shell', or 'bfs'.")  # positions for all nodes


    # Extract class labels (y attribute) for each node
    if any(G.nodes[node].get('y') is not None for node in G.nodes):
        # If y attribute is present, use it as labels
        node_labels = {node: G.nodes[node].get('y', '') for node in G.nodes}
        print("Node labels: ", node_labels)
        print("y attribute used for node labels")
        # Set node color based on y label
        y_values = [G.nodes[node].get('y', 0) for node in G.nodes]
        # Use a colormap for better visualization
        cmap = plt.cm.get_cmap('tab20', len(set(y_values)))
        node_color = [cmap(y) for y in y_values]
    else:
        node_labels = {node: str(node) for node in G.nodes()}
        node_color = "lightblue"

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_color)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="black")

    plt.title("Graph") # TODO: generic title
    plt.axis("off")

    if save_to_file:
        plt.savefig("graphs/community_graph.png", dpi=300)
        print("Graph saved to graphs/community_graph.png")
    plt.show()

    # torch_geometric.utils.visualize.draw(data, show=True, node_size=50, edge_width=1, font_size=10)

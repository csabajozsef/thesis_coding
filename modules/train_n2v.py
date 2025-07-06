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

from modules.utils import file_name_generator
from modules.utils import set_seeds_and_device

from modules.training import set_loader_and_optimizer
from modules.training import model_init
from modules.training import model_training_n2v
from modules.training import create_parameters_dict

import os
import time
import random
import json
import pickle
import copy
import os.path as osp
from pprint import pprint

device = set_seeds_and_device()
parameter_dicts = create_parameters_dict()

def train_node2vecs(data_list:list, num_training_epochs:int = 6, parameter_dicts = parameter_dicts, device = device):
    
    # TODO: for loop for parameter_dicts - prep for hyperparameter tuning - how to score?

    for data in data_list:
        for key, value in parameter_dicts.items():
            print(f"Running with {key} = {value}")
            # data = graph_lib.create_masks(data)
            model = model_init(value,data)
            
            loader, optimizer = set_loader_and_optimizer(model)
            num_training_epochs = num_training_epochs # Or 201, etc.

            best_state, training_history = model_training_n2v(
                model,
                value,
                data,
                loader,
                optimizer,
                num_training_epochs,
                device,
                model_save_path='../training_data/models/' # Example save path
            )
    # TODO: return results? what is the right format? pictures for sure

def train_node2vec(data_list:list, num_training_epochs:int = 6, parameter_dicts = parameter_dicts, device = device):
    
    # TODO: for loop for parameter_dicts - prep for hyperparameter tuning - how to score?

    for data in data_list:
        for key, value in parameter_dicts.items():
            print(f"Running with {key} = {value}")
            # data = graph_lib.create_masks(data)
            model = model_init(value,data)
            
            loader, optimizer = set_loader_and_optimizer(model)
            num_training_epochs = num_training_epochs # Or 201, etc.

            best_state, training_history = model_training_n2v(
                model,
                value,
                data,
                loader,
                optimizer,
                num_training_epochs,
                device,
                model_save_path='./training_data/models/' # Example save path
            )
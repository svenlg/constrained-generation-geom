import os

import random
import numpy as np
import torch

import dgl


def extract_trailing_numbers(run_name):
    import re
    match = re.search(r"(\d+)$", run_name)
    return int(match.group(1)) if match else None

def save(save_path, graphs):
    # save graphs and labels
    graph_path = os.path.join(save_path, 'dgl_graphs.bin')
    dgl.save_graphs(graph_path, graphs)

def load(save_path):
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(save_path, 'dgl_graph.bin')
    graphs, _ = dgl.load_graphs(graph_path)
    return graphs

# Update this function whenever you have a library that needs to be seeded.
def set_seed(seed: int):
    """Seed all random generators."""
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)

import os
from dgl import save_graphs, load_graphs

import random
import numpy as np
import torch


def extract_trailing_numbers(run_name):
    import re
    match = re.search(r"(\d+)$", run_name)
    return int(match.group(1)) if match else None

def save(save_path, graphs):
    # save graphs and labels
    graph_path = os.path.join(save_path, 'dgl_graphs.bin')
    save_graphs(graph_path, graphs)

def load(save_path):
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(save_path, 'dgl_graph.bin')
    graphs, _ = load_graphs(graph_path)
    return graphs

# Update this function whenever you have a library that needs to be seeded.
def seed_everything(seed: int):
    """Seed all random generators."""
    # For random:
    random.seed(seed)

    # For numpy:
    np.random.seed(seed)

    # For PyTorch:
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

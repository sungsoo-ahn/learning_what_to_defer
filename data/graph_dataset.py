import os
import random
import math
from dgl import DGLGraph
from torch.utils.data import Dataset
from data.util import read_dgl_from_metis

def generate_er_graph(n, p):
    G = DGLGraph()
    G.add_nodes(n)

    w = -1
    lp = math.log(1.0 - p)
    
    # Nodes in graph are from 0,n-1 (start with v as the first node index).
    v = 1
    edges_list = []
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr/lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            edges_list.extend([(v, w), (w, v)])
    
    G.add_edges(*zip(*edges_list))
    
    return G

class GraphDataset(Dataset):
    def __init__(
        self, 
        data_dir = None, 
        generate_fn = None,
        ):            
        self.data_dir = data_dir
        self.generate_fn = generate_fn
        if data_dir is not None:
            self.num_graphs = len([
                name 
                for name in os.listdir(data_dir)
                if name.endswith('.METIS')
                ])
        elif generate_fn is not None:
            self.num_graphs = 5000 # sufficiently large number for moving average
        else:
            assert False

    def __getitem__(self, idx):
        if self.generate_fn is None:
            g_path = os.path.join(
                self.data_dir, 
                "{:06d}.METIS".format(idx)
                )
            g = read_dgl_from_metis(g_path)
        else:
            g = self.generate_fn()

        return g
    
    def __len__(self):
        return self.num_graphs

def get_er_15_20_dataset(mode, data_dir = None):
    if mode == "train":
        def generate_fn():
            num_nodes = random.randint(15, 20)
            g = generate_er_graph(num_nodes, p = 0.15)
            return g
        
        return GraphDataset(generate_fn = generate_fn)
    else:    
        return GraphDataset(data_dir = data_dir)    
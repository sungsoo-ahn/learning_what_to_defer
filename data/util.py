import networkx as nx
import dgl
import json

def read_dgl_from_metis(metis_path):
    edges_set = set()
    with open(metis_path, "r") as f:
        lines = f.readlines()
        num_nodes, num_edges = list(map(int, lines[0].split()))

        for u, line in enumerate(lines[1:]):
            nums = list(map(int, line.split()))
            for v in nums:
                edges_set.add((u, v))
                edges_set.add((v, u))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(*zip(*edges_set))
    return g

def write_nx_to_metis(g, path): 
    with open(path, "w") as f:
        # write the header
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        f.write("{} {}\n".format(num_nodes, num_edges))
        
        # now write all edges
        for u in g.nodes():
            sorted_adj_nodes = sorted([v for v in g[u]])
            neighbors = " ".join([str(v) for v in sorted_adj_nodes])
            f.write("{}\n".format(neighbors))
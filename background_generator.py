import numpy as np
import networkx as nx

def generate():
    base_g = nx.Graph()
    base_g.add_node(0, pos=[0, 0, 0])

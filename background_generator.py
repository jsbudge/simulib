import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def generate(gx, gy, gz, rg):
    base_g = nx.Graph()
    base_g.add_node(0, pos=[0, 0, 0])
    base_g.add_node(1, pos=[1, 0, 0])
    base_g.add_node(2, pos=[0, 1, 0])
    return base_g


if __name__ == '__main__':
    b = generate()
    plt.figure()
    nx.draw(b)
    plt.show()

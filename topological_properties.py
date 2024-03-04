import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

file_name = "SFHH.xlsx"


def read_file():
    dataframe = pd.read_excel(file_name)
    edges = set()
    nodes = set()
    for index, row in dataframe.iterrows():
        edge = (row['node1'], row['node2'])
        nodes.add(row['node1'])
        nodes.add(row['node2'])
        edges.add(edge)

    return edges, nodes


def get_average_degree(network: ig.Graph):
    degrees = network.degree()
    avg_degree = sum(degrees) / len(degrees)
    return avg_degree


if __name__ == "__main__":
    edges, nodes = read_file()
    network = ig.Graph(len(nodes), edges)

    # PART A.1
    print('Number of nodes: ', len(nodes))
    print('Number of links: ', len(edges))
    print('Average degree: ', get_average_degree(network))
    print('Standard deviation: ', np.std(np.array(network.degree())))

    # PART A.2


import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

file_name = "SFHH.xlsx"


def read_file():
    dataframe = pd.read_excel(file_name)
    edges = []
    nodes = set()
    for index, row in dataframe.iterrows():
        edge = (row['node1'], row['node2'])
        nodes.add(row['node1'])
        nodes.add(row['node2'])
        edges.append(edge)

    return edges, nodes


if __name__ == "__main__":
    edges, nodes = read_file()
    network = ig.Graph(len(nodes), edges)
    degrees = network.degree()

    # PART A.1
    print('Number of nodes: ', len(nodes))  # 403
    print('Number of links: ', len(edges))  # 70261
    print('Average degree: ', sum(degrees) / len(degrees))  # 347.8267326732673
    print('Standard deviation: ', np.std(np.array(network.degree())))  # 377.2790805649033

    # PART A.2
    degrees = network.degree()
    degree_frequency = {}
    for degree in degrees:
        if degree in degree_frequency:
            degree_frequency[degree] += 1
        else:
            degree_frequency[degree] = 1
    degree_frequency = dict(sorted(degree_frequency.items()))
    for key, value in degree_frequency.items():
        degree_frequency[key] = float(value/len(nodes))
    plt.plot(list(degree_frequency.keys()), list(degree_frequency.values()), marker='^', linestyle='-')
    plt.xlabel('Degree')
    plt.ylabel('P(degree)')
    plt.title('Degree Distribution')
    plt.savefig('a_2.png')
    plt.show()

    # PART A.3
    print('Degree correlation (assortativity): ', network.assortativity_degree())  # 0.4126192209888443

    # PART A.4
    print('Clustering coefficient: ', network.transitivity_undirected())  # 0.23590263303822398

    # PART A.5
    print('Average hop count: ', network.average_path_length())  # 1.9530140858980531
    print('Diameter : ', network.diameter())  # 4

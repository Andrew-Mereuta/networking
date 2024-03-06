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
        if not (edge in edges) and not ((row['node2'], row['node1']) in edges):
            edges.add(edge)
        nodes.add(row['node1'])
        nodes.add(row['node2'])

    return edges, nodes


def plot_probability(dictionary):
    frequencies = {}
    for _, weight in dictionary.items():
        if weight in frequencies:
            frequencies[weight] += 1
        else:
            frequencies[weight] = 1

    total_count = sum(frequencies.values())
    probabilities = {weight: count / total_count for weight, count in frequencies.items()}
    sorted_probabilities = sorted(probabilities.items())
    weight_values, probabilities = zip(*sorted_probabilities)

    plt.plot(weight_values, probabilities, marker='o', linestyle='-')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Weight values, log scaled')
    plt.ylabel('Probability, log scaled')
    plt.title('Probability Distribution')
    plt.savefig('a_7.png')
    plt.show()


def calculate_degree(edges, nodes):
    degrees = {node: 0 for node in nodes}
    used_edges = set()
    for (n1, n2) in edges:
        if not ((n1, n2) in used_edges or (n2, n1) in used_edges):
            degrees[n1] += 1
            degrees[n2] += 1
            used_edges.add((n1, n2))
            used_edges.add((n2, n1))

    return sorted(nodes, key=lambda x: degrees[x], reverse=True)


def get_weight_by_edge():
    dataframe = pd.read_excel(file_name)
    weight_by_edge = {}
    for index, row in dataframe.iterrows():
        edge = (row['node1'], row['node2'])
        if edge in weight_by_edge:
            weight_by_edge[edge] += 1
        elif (row['node2'], row['node1']) in weight_by_edge:
            weight_by_edge[(row['node2'], row['node1'])] += 1
        else:
            weight_by_edge[edge] = 1

    return weight_by_edge


if __name__ == "__main__":
    edges, nodes = read_file()
    network = ig.Graph(len(nodes), edges)
    degrees = network.degree()[1:]

    # PART A.1
    print('Number of nodes: ', len(nodes))  # 403
    print('Number of links: ', len(edges))  # 70261
    print('Average degree: ', sum(degrees) / len(degrees))  # 347.8267326732673
    print('Standard deviation: ', np.std(np.array(degrees)))  # 377.2790805649033

    # PART A.2
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

    # PART A.7
    weight_by_edge = get_weight_by_edge()
    plot_probability(weight_by_edge)

import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

file_name = "SFHH.xlsx"


def read_file():
    dataframe = pd.read_excel(file_name)
    nodes = set()
    edges_by_timestamp = {}
    edges = list()
    for index, row in dataframe.iterrows():
        edge = (row['node1'], row['node2'])
        nodes.add(row['node1'])
        nodes.add(row['node2'])
        timestamp = row['timestamp']
        if timestamp in edges_by_timestamp:
            edges_by_timestamp[timestamp].append(edge)
        else:
            edges_by_timestamp[timestamp] = [edge]
        edge = (row['node1'], row['node2'])
        edges.append(edge)

    return dict(sorted(edges_by_timestamp.items())), nodes, edges


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


def calculate_weight_node(edges, nodes):
    weight_by_edge = {}
    for edge in edges:
        (n1, n2) = edge
        if edge in weight_by_edge:
            weight_by_edge[edge] = weight_by_edge[edge] + 1
        elif (n2, n1) in weight_by_edge:
            weight_by_edge[(n2, n1)] = weight_by_edge[(n2, n1)] + 1
        else:
            weight_by_edge[edge] = 1

    node_strength = {node: 0 for node in nodes}
    for node in nodes:
        for edge, weight in weight_by_edge.items():
            if node in edge:
                node_strength[node] += weight
    return sorted(nodes, key=lambda x: node_strength[x], reverse=True)


def first_contact_in_network(edges_by_timestamp):
    first_contact = {}
    for (timestamp, edges) in edges_by_timestamp.items():
        for (n1, n2) in edges:
            if n1 in first_contact:
                first_contact[n1] = min(timestamp, first_contact[n1])
            else:
                first_contact[n1] = timestamp
            if n2 in first_contact:
                first_contact[n2] = min(timestamp, first_contact[n2])
            else:
                first_contact[n2] = timestamp
    return dict(sorted(first_contact.items(), key=lambda x: x[1]))


def find_first_timestamp_and_link(edges_by_timestamp, seed):
    for timestamp, edges in edges_by_timestamp.items():
        for (n1, n2) in edges:
            if seed == n1 or seed == n2:
                return timestamp, (n1, n2)


def infect(edges_by_timestamp: dict, infected: set, timestamp: int):
    infected_nodes = infected.copy()
    infected_links = []
    for (n1, n2) in edges_by_timestamp[timestamp]:
        if n1 in infected or n2 in infected:
            infected_nodes.add(n1)
            infected_nodes.add(n2)
            infected_links.append((n1, n2))

    return infected_nodes, infected_links


def get_networks(edges_by_timestamp: dict, nodes: list[int]):
    networks = []
    infected_nodes_by_timestamp = {}
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp, link = find_first_timestamp_and_link(edges_by_timestamp, seed)
        infected_links = [link]
        # TODO: check boundary
        for timestamp in range(first_timestamp, len(edges_by_timestamp)):
            infected_ns, infected_ls = infect(edges_by_timestamp, infected_nodes, timestamp)
            infected_nodes = infected_ns.union(infected_nodes)
            infected_links.extend(infected_ls)
            if timestamp in infected_nodes_by_timestamp:
                infected_nodes_by_timestamp[timestamp].append(len(infected_nodes))
            else:
                infected_nodes_by_timestamp[timestamp] = [len(infected_nodes)]
            if seed not in infected_nodes_by_seed:
                if len(infected_nodes) > 0.8 * len(nodes):
                    infected_nodes_by_seed[seed] = timestamp

        networks.append(ig.Graph(infected_links))

    return networks, infected_nodes_by_timestamp, dict(sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))


def calculate_average_infected(infected_nodes_by_timestamp):
    average_infected = {}
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        num_networks = len(num_infected_list)
        total_infected = sum(num_infected_list)
        average_infected[timestamp] = total_infected / num_networks
    return average_infected


def calculate_standard_deviation(infected_nodes_by_timestamp):
    std_deviation_by_timestamp = {}
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        std_deviation = np.std(num_infected_list)
        std_deviation_by_timestamp[timestamp] = std_deviation
    return std_deviation_by_timestamp


def plot_average_infected_with_errorbars(infected_nodes_by_timestamp):
    timestamps = list(infected_nodes_by_timestamp.keys())
    averages = []
    std_deviations = []
    for timestamp, num_infected_list in infected_nodes_by_timestamp.items():
        average = np.mean(num_infected_list)
        std_deviation = np.std(num_infected_list)
        averages.append(average)
        std_deviations.append(std_deviation)

    # Plot
    plt.errorbar(timestamps, averages, yerr=std_deviations, fmt='o-', label='Average Infected')
    plt.xlabel('Timestamp')
    plt.ylabel('Average Number of Infected Nodes (E[I(t)])')
    plt.title('Average Number of Infected Nodes Over Time')
    plt.savefig('b_8.png')
    plt.legend()
    plt.show()


# b9
def plot_infected_nodes_by_seed(sorted_infected_nodes):

    seeds = list(sorted_infected_nodes.keys())
    timestamps = list(sorted_infected_nodes.values())

    plt.figure(figsize=(10, 6))
    plt.plot(seeds, timestamps, marker='o', linestyle='-', color='b')
    plt.xlabel('Seed Nodes')
    plt.ylabel('Timestamp to Reach Goal')
    plt.title('Timestamp for Nodes to Reach Goal for Each Seed')
    num_ticks = 50
    if len(seeds) > num_ticks:
        step = len(seeds) // num_ticks
        plt.xticks(seeds[::step], rotation=45, fontsize=5)
    else:
        plt.xticks(seeds, rotation=45, fontsize=5)

    plt.tight_layout()
    plt.show()

#b10
def centrality(nodes, edges, sorted_infected_nodes, edges_by_timestamp):
    sorted_infected_nodes = list(sorted_infected_nodes.keys())
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    degrees = calculate_degree(edges, nodes)
    weights = calculate_weight_node(edges, nodes)
    fc = list(first_contact_in_network(edges_by_timestamp).keys())
    rRD_values = []
    rRS_values = []
    rFS_values = []
    for v in f:
        fraction = int(v * len(nodes))
        Rf = set(sorted_infected_nodes[0:fraction])
        Rf_degree = set(degrees[0:fraction])
        Rf_strength = set(weights[0:fraction])
        Rf_contact = set(fc[0:fraction])
        rRD = len(Rf.intersection(Rf_degree))/len(Rf)
        rRS = len(Rf.intersection(Rf_strength))/len(Rf)
        rFS = len(Rf.intersection(Rf_contact))/len(Rf)

        rRD_values.append(rRD)
        rRS_values.append(rRS)
        rFS_values.append(rFS)

    plt.figure(figsize=(10, 6))
    plt.plot(f, rRD_values, marker='o', label='rRD')
    plt.plot(f, rRS_values, marker='s', label='rRS')
    plt.plot(f, rFS_values, marker='^', label='rFS')
    plt.xlabel('f')
    plt.ylabel('Recognition Rate')
    plt.title('Recognition Rate vs. f')
    plt.xticks(f)
    plt.legend()
    plt.savefig('b_10_11.png')
    plt.show()



if __name__ == "__main__":
    edges_by_timestamp, nodes, edges = read_file()
    networks, infected_nodes_by_timestamp, sorted_infected_nodes = get_networks(edges_by_timestamp, nodes)
    print(calculate_average_infected(infected_nodes_by_timestamp))

    print(len(networks))
    plot_average_infected_with_errorbars(infected_nodes_by_timestamp)

    plot_infected_nodes_by_seed(sorted_infected_nodes)

    centrality(nodes, edges, sorted_infected_nodes, edges_by_timestamp)

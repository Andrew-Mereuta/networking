import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt

file_name = "SFHH.xlsx"


def read_file():
    dataframe = pd.read_excel(file_name)
    nodes = set()
    edges_by_timestamp = {}

    for index, row in dataframe.iterrows():
        edge = (row['node1'], row['node2'])
        nodes.add(row['node1'])
        nodes.add(row['node2'])
        timestamp = row['timestamp']
        if timestamp in edges_by_timestamp:
            edges_by_timestamp[timestamp].append(edge)
        else:
            edges_by_timestamp[timestamp] = [edge]

    return dict(sorted(edges_by_timestamp.items())), nodes


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

        networks.append(ig.Graph(infected_links))

    return networks, infected_nodes_by_timestamp


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


if __name__ == "__main__":
    edges_by_timestamp, nodes = read_file()
    networks, infected_nodes_by_timestamp = get_networks(edges_by_timestamp, nodes)
    print(calculate_average_infected(infected_nodes_by_timestamp))


    print(len(networks))
    plot_average_infected_with_errorbars(infected_nodes_by_timestamp)


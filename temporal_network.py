import random

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
    sorted_grouped = {}
    degrees = dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True))
    for key, val in sorted(degrees.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0], reverse=True))


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
    sorted_grouped = {}
    degrees = dict(sorted(node_strength.items(), key=lambda item: item[1], reverse=True))
    for key, val in sorted(degrees.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0], reverse=True))


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
    first_contact = dict(sorted(first_contact.items(), key=lambda item: item[1], reverse=True))
    sorted_grouped = {}
    for key, val in sorted(first_contact.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return dict(sorted(sorted_grouped.items(), key=lambda x: x[0]))


def find_first_timestamp_and_link(edges_by_timestamp, seed):
    for timestamp, edgess in edges_by_timestamp.items():
        for (n1, n2) in edgess:
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


def get_networks(edges_by_timestamp: dict, nodes: list[int], infection_goal):
    networks = []
    infected_nodes_by_timestamp = {}
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp, link = find_first_timestamp_and_link(edges_by_timestamp, seed)
        infected_links = [link]
        # TODO: check boundary
        for timestamp in range(first_timestamp, len(edges_by_timestamp.keys())):
            infected_ns, infected_ls = infect(edges_by_timestamp, infected_nodes, timestamp)
            infected_nodes = infected_ns.union(infected_nodes)
            infected_links.extend(infected_ls)
            if timestamp in infected_nodes_by_timestamp:
                infected_nodes_by_timestamp[timestamp].append(len(infected_nodes))
            else:
                infected_nodes_by_timestamp[timestamp] = [len(infected_nodes)]
            if seed not in infected_nodes_by_seed:
                if len(infected_nodes) > infection_goal * len(nodes):
                    infected_nodes_by_seed[seed] = timestamp

        networks.append(ig.Graph(infected_links))
    sorted_grouped = {}
    # infected_nodes_by_seed = dict(sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))
    for key, val in sorted(infected_nodes_by_seed.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return networks, infected_nodes_by_timestamp, dict(sorted(sorted_grouped.items(), key=lambda x: x[0])),  dict(sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))

def get_networksb12(edges_by_timestamp: dict, nodes: list[int], infection_goal):
    networks = []
    infected_nodes_by_timestamp = {}
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp, link = find_first_timestamp_and_link(edges_by_timestamp, seed)
        infected_links = [link]
        # TODO: check boundary
        for timestamp in range(first_timestamp, len(edges_by_timestamp.keys())):
            infected_ns, infected_ls = infect(edges_by_timestamp, infected_nodes, timestamp)
            infected_nodes = infected_ns.union(infected_nodes)
            infected_links.extend(infected_ls)
            if timestamp in infected_nodes_by_timestamp:
                infected_nodes_by_timestamp[timestamp].append(len(infected_nodes))
            else:
                infected_nodes_by_timestamp[timestamp] = [len(infected_nodes)]
            if len(infected_nodes) <= infection_goal * len(nodes):
                if seed in infected_nodes_by_seed:
                    infected_nodes_by_seed[seed].append(timestamp)
                else:
                    infected_nodes_by_seed[seed] = [(timestamp)]

        networks.append(ig.Graph(infected_links))
    average_times = {}
    for seed, times in infected_nodes_by_seed.items():
        average_times[seed] = sum(times) / len(times)
    sorted_grouped = {}
    # infected_nodes_by_seed = dict(sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))
    for key, val in sorted(average_times.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]
    return networks, infected_nodes_by_timestamp, dict(sorted(sorted_grouped.items(), key=lambda x: x[0])),  dict(sorted(average_times.items(), key=lambda item: item[1]))


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
        average = calculate_average_infected(infected_nodes_by_timestamp)
        std_deviation = np.std(num_infected_list)
        averages.append(average)
        std_deviations.append(std_deviation)
    averages = list(calculate_average_infected(infected_nodes_by_timestamp).values())
    std_deviations = list(calculate_standard_deviation(infected_nodes_by_timestamp).values())
    plt.errorbar(timestamps, averages, yerr=std_deviations)
    plt.xlabel('Timestamp')
    plt.ylabel('Average Number of Infected Nodes (E[I(t)])')
    plt.title('Average Number of Infected Nodes Over Time')
    plt.savefig('b_8.png')
    plt.show()


# b9
def plot_infected_nodes_by_seed(sorted_infected_nodes, assignment_num):

    seeds = list(sorted_infected_nodes.keys())
    timestamps = list(sorted_infected_nodes.values())

    plt.figure(figsize=(10, 6))
    plt.plot(seeds, timestamps, marker='o', color='b')
    plt.xlabel('Seed Nodes')
    plt.ylabel('Timestamp to Reach Goal')
    plt.title('Timestamp for Nodes to Reach Goal for Each Seed')
    num_ticks = 50
    if len(seeds) > num_ticks:
        step = 10
        plt.xticks(seeds[::step], rotation=45, fontsize=8)
    else:
        plt.xticks(seeds, rotation=45, fontsize=5)

    plt.tight_layout()
    plt.savefig(f"b_{assignment_num}.png")
    plt.show()

#b10
def centrality(nodes, edges, sorted_infected_nodes, edges_by_timestamp, num):
    # sorted_infected_nodes = list(sorted_infected_nodes.keys())
    f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    degrees = calculate_degree(edges, nodes)
    weights = calculate_weight_node(edges, nodes)
    fc = first_contact_in_network(edges_by_timestamp)
    rRD_values = []
    rRS_values = []
    rFS_values = []
    for v in f:
        fraction = int(v * len(nodes))
        rRD = 0
        rRS = 0
        rFS = 0
        for _ in range(1000):
            Rf_degree = set()
            Rf_strength = set()
            Rf = set()
            Rf_contact = set()
            degreess = list(degrees.items())[0:int(v * len(nodes))]
            for key, value in degreess:
                Rf_degree.add(random.choice(value))
            weightss = list(weights.items())[0:int(v * len(nodes))]
            for key, value in weightss:
                Rf_strength.add(random.choice(value))
            sorted_infected_nodess = list(sorted_infected_nodes.items())[0:int(v * len(nodes))]
            for key, value in sorted_infected_nodess:
                Rf.add(random.choice(value))
            fcs = list(fc.items())[0:int(v * len(nodes))]
            for key, value in fcs:
                Rf_contact.add(random.choice(value))
            rRD += len(Rf.intersection(Rf_degree))/len(Rf)
            rRS += len(Rf.intersection(Rf_strength))/len(Rf)
            rFS += len(Rf.intersection(Rf_contact))/len(Rf)
        rRD /= 1000
        rRS /= 1000
        rFS /= 1000
        rRD_values.append(rRD)
        rRS_values.append(rRS)
        rFS_values.append(rFS)

    plt.figure(figsize=(10, 6))
    plt.plot(f, rRD_values, marker='o', label='rRD')
    plt.plot(f, rRS_values, marker='s', label='rRS')
    if num == "10":
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for degree and strength vs fraction')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_10.png')
        plt.show()
    else:
        plt.plot(f, rFS_values, marker='^', label='r_first_contact')
        plt.xlabel('Fraction according to nodes influence')
        plt.ylabel('Recognition Rate')
        plt.title('Recognition Rate for degree, strength and first contact vs fraction')
        plt.xticks(f)
        plt.legend()
        plt.savefig('b_11.png')
        plt.show()

def get_networks_b12(edges_by_timestamp: dict, nodes: list[int], infection_goal):
    networks = []
    infected_nodes_by_timestamp = {}
    infected_nodes_by_seed = {}
    for seed in nodes:
        infected_nodes = set()
        infected_nodes.add(seed)
        first_timestamp, link = find_first_timestamp_and_link(edges_by_timestamp, seed)
        infected_links = [link]
        total_time = 0
        # TODO: check boundary
        for timestamp in range(first_timestamp, len(edges_by_timestamp)):
            old_num_infected_nodes = len(infected_nodes)
            infected_ns, infected_ls = infect(edges_by_timestamp, infected_nodes, timestamp)
            infected_nodes = infected_ns.union(infected_nodes)
            infected_links.extend(infected_ls)
            
            total_time += timestamp * (len(infected_nodes) - old_num_infected_nodes)

            if timestamp in infected_nodes_by_timestamp:
                infected_nodes_by_timestamp[timestamp].append(len(infected_nodes))
            else:
                infected_nodes_by_timestamp[timestamp] = [len(infected_nodes)]
            if seed not in infected_nodes_by_seed:
                if len(infected_nodes) > infection_goal * len(nodes):
                    infected_nodes_by_seed[seed] = total_time/(infection_goal*len(nodes))

        networks.append(ig.Graph(infected_links))

    sorted_grouped = {}
    for key, val in sorted(infected_nodes_by_seed.items()):
        if val in sorted_grouped:
            sorted_grouped[val].append(key)
        else:
            sorted_grouped[val] = [key]

    return networks, infected_nodes_by_timestamp, dict(sorted(sorted_grouped.items(), key=lambda x: x[0])), dict(sorted(infected_nodes_by_seed.items(), key=lambda item: item[1]))


# b12
def plot_different_infection_metrics(sorted_infected_nodes, assignment_num):

    seeds = list(sorted_infected_nodes.keys())
    timestamps = list(sorted_infected_nodes.values())
    
    plt.figure(figsize=(10, 6))
    for method_name, infected_nodes in sorted_infected_nodes.items():
        print(method_name)
        seeds = list(infected_nodes.keys())
        timestamps = list(infected_nodes.values())   
        plt.plot(seeds, timestamps, marker='o', label=method_name)
        plt.xlabel('Seed Nodes')
        plt.ylabel('Timestamp to Reach Goal')
    plt.title('Timestamp for Nodes to Reach Goal for Each Seed')
    num_ticks = 50
    if len(seeds) > num_ticks:
        step = 10
        plt.xticks(seeds[::step], rotation=45, fontsize=8)
    else:
        plt.xticks(seeds, rotation=45, fontsize=5)

    plt.tight_layout()
    plt.savefig(f"b_{assignment_num}.png")
    plt.show()

def b12(sorted_infected_nodes_r, sorted_infected_nodes_r_star, sorted_infected_nodes_r_accent, num):
        f = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        rRD_values = []
        rRS_values = []
        rFS_values = []
        for v in f:
            fraction = int(v * len(nodes))
            rRD = 0
            rRS = 0
            rFS = 0
            for _ in range(1000):
                Rf_star = set()
                Rf_accent = set()
                Rf = set()
                degreess = list(sorted_infected_nodes_r.items())[0:int(v * len(nodes))]
                for key, value in degreess:
                    Rf.add(random.choice(value))
                weightss = list(sorted_infected_nodes_r_star.items())[0:int(v * len(nodes))]
                for key, value in weightss:
                    Rf_star.add(random.choice(value))
                sorted_infected_nodess = list(sorted_infected_nodes_r_accent.items())[0:int(v * len(nodes))]
                for key, value in sorted_infected_nodess:
                    Rf_accent.add(random.choice(value))
                rRD += len(Rf.intersection(Rf_star)) / len(Rf)
                rRS += len(Rf.intersection(Rf_accent)) / len(Rf)
            rRD /= 1000
            rRS /= 1000
            rFS /= 1000
            rRD_values.append(rRD)
            rRS_values.append(rRS)
            rFS_values.append(rFS)

        plt.figure(figsize=(10, 6))
        if num == '1':
            plt.plot(f, rRD_values, marker='o', label='rf star')
            plt.plot(f, rRS_values, marker='s', label='rf accent')
            plt.xlabel('Fraction according to nodes influence')
            plt.ylabel('Recognition Rate')
            plt.title('Recognition Rate for rf* and rf\' for r')
            plt.xticks(f)
            plt.legend()
            plt.savefig('b_12_1.png')
            plt.show()
        if num == '2':
            plt.plot(f, rRD_values, marker='o', label='rf usual')
            plt.plot(f, rRS_values, marker='s', label='rf star')
            plt.xlabel('Fraction according to nodes influence')
            plt.ylabel('Recognition Rate')
            plt.title('Recognition Rate for rf and rf* for r accent')
            plt.xticks(f)
            plt.legend()
            plt.savefig('b_12_2.png')
            plt.show()
        if num == '3':
            plt.plot(f, rRD_values, marker='o', label='rf usual')
            plt.plot(f, rRS_values, marker='s', label='rf accent')
            plt.xlabel('Fraction according to nodes influence')
            plt.ylabel('Recognition Rate')
            plt.title('Recognition Rate for rf and rf\' for r star')
            plt.xticks(f)
            plt.legend()
            plt.savefig('b_12_3.png')
            plt.show()

if __name__ == "__main__":
    edges_by_timestamp, nodes, edges = read_file()
    networks, infected_nodes_by_timestamp, sorted_infected_nodes, not_map = get_networks(edges_by_timestamp, nodes, 0.8)

    # PART B.8
    # plot_average_infected_with_errorbars(infected_nodes_by_timestamp)

    # PART B.9
    # plot_infected_nodes_by_seed(not_map, "9")

    # # PART B.10
    # centrality(nodes, edges, sorted_infected_nodes, edges_by_timestamp, "10")
    #
    # # PART B.11
    # centrality(nodes, edges, sorted_infected_nodes, edges_by_timestamp, "11")

    # # For 12 we need to rank on 10% reached.
    # networks_r, infected_nodes_by_timestamp_r, sorted_infected_nodes_r, not_map_r = get_networks(edges_by_timestamp, nodes, 0.8)
    networks_r_star, infected_nodes_by_timestamp_r_star, sorted_infected_nodes_r_star, not_map_r_star = get_networks(edges_by_timestamp, nodes, 0.1)
    networks_r_accent, infected_nodes_by_timestamp_r_accent, sorted_infected_nodes_r_accent, not_map_r_star = get_networksb12(edges_by_timestamp, nodes, 0.8)
    # print(f"len r = {len(not_map_r)}, len r* = {len(not_map_r_star)}, len r'= {len(sorted_infected_nodes_r_accent)}")
    # infection_metrics = {
    #     "R": not_map_r,
    #     "R*": not_map_r_star,
    #     "R'": sorted_infected_nodes_r_accent
    # }
    # plot_different_infection_metrics(infection_metrics, "12")
    b12(sorted_infected_nodes, sorted_infected_nodes_r_star, sorted_infected_nodes_r_accent, '1')
    b12(sorted_infected_nodes_r_accent, sorted_infected_nodes, sorted_infected_nodes_r_star, '2')
    b12(sorted_infected_nodes_r_star, sorted_infected_nodes, sorted_infected_nodes_r_accent, '3')







import pandas as pd
from community import community_louvain
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_graph():
    """
    Generate a graph from facebook_combined dataset file.
    returns: A Graph with nodes and edges
    """
    fb_network_data = pd.read_table('facebook_combined.txt')
    edges = []
    for column in fb_network_data:
        for i in fb_network_data[column].values:
            edges.append(tuple(map(int, i.split())))

    fb_graph = nx.Graph()
    fb_graph.add_edges_from(edges)
    return fb_graph


def partition_graph():
    g = generate_graph()
    # return partition as a dict
    partition = community_louvain.best_partition(g)
    return partition


def visualize_network():
    g = generate_graph()
    partition = partition_graph()

    # draw the graph
    pos = nx.spring_layout(g)

    # color the nodes according to their partition
    color_map = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(g, pos, partition.keys(), node_size=40,
                           cmap=color_map, node_color=list(partition.values()))
    nx.draw_networkx_edges(g, pos, alpha=0.5)
    plt.show()

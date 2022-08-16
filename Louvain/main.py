import sys
import pandas as pd
from community import community_louvain
import networkx as nx
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import timeit

"""
Timing code from line 13 - 34 adopted 
from https://piazza.com/northeastern/spring2022/ds5010bemis/resources - Lecture Code >> timing.py
"""
if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 1000

if len(sys.argv) > 2:
    reps = int(sys.argv[2])
else:
    reps = 10


def bench(expr="pass", reps=100, echo=False):
    elapsed = timeit.timeit(expr, globals=globals(), number=reps)
    time = elapsed / reps
    if echo:
        t = format(time, ".9f")
        s = "Average runtime of " + expr + ": " + t + "s (avg of " + str(reps) + " reps)"
        print(s)
    return time


echo = True


def generate_graph():
    """
    Generate a graph from facebook_combined dataset file.
    returns: A Graph with nodes and edges
    """
    fb_network_data = pd.read_table('Louvain/facebook_combined.txt')
    edges = []
    for column in fb_network_data:
        for i in fb_network_data[column].values:
            edges.append(tuple(map(int, i.split())))

    fb_graph = nx.Graph()
    fb_graph.add_edges_from(edges)
    return fb_graph


def partition_graph():
    """
    Get the partition of the graph nodes which maximises the modularity using the Louvain heuristices
    returns: A dictionary where the keys are nodes and values are communities numbered from 0 to number of communities
    """
    g = generate_graph()
    partition = community_louvain.best_partition(g)
    return partition


def communities_count():
    """
    returns: Integer, number of communities detected in the network.
    """
    g = generate_graph()
    # A list of sets (partition of g). Each set represents one community and contains all the nodes that constitute it.
    communities_list = nx_comm.louvain_communities(g)
    return len(communities_list)


def graph_modularity():
    """
    Get the modularity of a partition of a graph  using the community.modularity(partition, graph, weight='weight') method
    returns: A float
    """
    g = generate_graph()
    p = partition_graph()
    return community_louvain.modularity(p, g)


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
    plt.show(block=True)


if __name__ == "__main__":
    print("\n")
    print("Number of communities detected:", communities_count())
    bench('partition_graph()', reps, echo)
    print("Network Modularity:", graph_modularity())
    print("Communities detected network graph loading ... Please wait for about 3 minutes.")
    visualize_network()

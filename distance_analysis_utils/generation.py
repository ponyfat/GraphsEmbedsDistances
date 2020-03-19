import random
import os
import networkx as nx
from functools import partial
import numpy as np
from tqdm import tqdm_notebook
import scipy

def generateRandomGNMs(d, n, m):
    graphs = []
    for i in range(0, d):
        graph = nx.gnm_random_graph(n, m)
        graphs.append(graph)
    return graphs

def generateComplicatedClicks(d=10):
    clicks = []
    for i in range(1, d + 1):
        click = nx.complete_graph(i)
        click.add_nodes_from(list(np.arange(i, d)))
        click.add_edges_from([(0, i) for i in np.arange(i, d)])
        clicks.append(click)
    return clicks

def generateClicks(d=10):
    clicks = []
    for i in range(1, d + 1):
        click = nx.complete_graph(i)
        click.add_nodes_from(list(np.arange(i, d)))
        clicks.append(click)
    return clicks

def randomEdge(G):
    return random.choice(list(G.edges))

def removeRandomEdge(G):
    removed_edge = randomEdge(G)
    G.remove_edge(removed_edge[0], removed_edge[1])
    return G

def generateRandomEdgeRemoval(d=10):
    rer = [nx.complete_graph(d)]
    for i in tqdm_notebook(range(0, d * (d - 1) // 2)):
        next_graph = removeRandomEdge(rer[-1].copy())
        rer.append(next_graph)
    return rer

def generateDiffSizedClicks(d=10):
    clicks = []
    for i in range(1, d + 1):
        click = nx.complete_graph(i)
        clicks.append(click)
    return clicks


def dumpDataset(dataset_name, dataset, lxml=False):
    # add a folder for the dataset
    try:
        os.mkdir(os.getcwd() + '/' + dataset_name)
    except Exception:
        return
    # dump all graphml files
    for i, graph in enumerate(dataset):
        if lxml:
            with open(f"{dataset_name}/{dataset_name}_{i + 1}.graphml", 'w') as f:
                nx.write_graphml_lxml(graph, f"{dataset_name}/{dataset_name}_{i + 1}.graphml")
        else:
            with open(f"{dataset_name}/{dataset_name}_{i + 1}.graphml", 'w') as f:
                nx.write_graphml_xml(graph, f"{dataset_name}/{dataset_name}_{i + 1}.graphml")

def createDataset(generator, name):
    ds = generator()
    dumpDataset(name, ds)

def createClicks(d=10):
    createDataset(partial(generateClicks, d), f'Clicks{d}')

def createRER(d=10):
    createDataset(partial(generateRandomEdgeRemoval, d), f'RER{d}')

def createGNM(d=10, n=10, m=5):
    createDataset(partial(generateRandomGNMs, d, n, m), f'GNM{d}')
from itertools import combinations
from functools import reduce
import random
import os
import networkx as nx
from functools import partial
import numpy as np
from tqdm import tqdm_notebook
import scipy
from distance_analysis_utils.analysis import *

class GraphletEmbeddings():
    def powerset(s):
        ps = lambda s: reduce(lambda P, x: P + [subset | {x} for subset in P], s, [set()])
        return ps(s)

    def __init__(self):
        pass
    
    def compute_graflets_set(self, k):
        print('Computing graflets...')
        all_possible_edges = set(combinations(np.arange(0, k), 2))
        all_possible_subgraphs_edges = GraphletEmbeddings.powerset(all_possible_edges)
        
        graphlets = []
        for subgraph_edges in tqdm_notebook(all_possible_subgraphs_edges):
            graphlet = nx.Graph()
            graphlet.add_nodes_from(np.arange(0, k))
            graphlet.add_edges_from(subgraph_edges)
            is_isomorphic = np.array([nx.is_isomorphic(other_graphlet, graphlet) for other_graphlet in graphlets]).sum()
            if is_isomorphic > 0:
                continue
            graphlets.append(graphlet)
        return graphlets
        
    def compute_embedding(self, G, k, graphlets):
        n = G.number_of_nodes()
        embed = np.zeros(len(graphlets))
        for nodes_subset in combinations(np.arange(0, n), k):
            subgraph = nx.subgraph(G, nodes_subset)
            for i, graphlet in enumerate(graphlets):
                if nx.is_isomorphic(graphlet, subgraph):
                    embed[i] += 1
                    break
        return embed
    
    def compute_embeddings(self, Gs, k):
        graphlets = self.compute_graflets_set(k)
        embeddings = []
        print('Computing embeddings...')
        for G in tqdm_notebook(Gs):
            e = self.compute_embedding(G, k, graphlets)
            embeddings.append(e)
        return np.array(embeddings)

class WLEmbeddings():
    def get_embedding(labels, unique_labels_to_inds, embedding_size):
        unique, counts = np.unique(labels, return_counts=True)
        embedding = np.zeros(embedding_size)
        for label, count in zip(unique, counts):
            ind = unique_labels_to_inds[label]
            embedding[ind] = count
        return embedding
    
    def __init__(self):
        pass
    
    def relabel_graph(self, labels, G):
        new_labels = []
        for node in range(G.number_of_nodes()):
            neighbors = list(G.neighbors(node))
            neighbors.append(node)
            
            neighbors_labels = labels[neighbors]
            new_label = tuple(neighbors_labels)
            new_labels.append(hash(new_label))
        return np.array(new_labels)
            
    def compute_labels(self, G, iterations):
        n = G.number_of_nodes()
        labels = np.zeros(n)
        for i in range(iterations):
            labels = self.relabel_graph(labels, G)
        return labels
    
    def compute_embeddings(self, Gs, iterations):
        print('Computing embeddings...')
        graphs_labels = []
        for G in tqdm_notebook(Gs):
            labels = self.compute_labels(G, iterations)
            graphs_labels.append(labels)
       
        # computing embeddings from labels
        unique_labels = np.concatenate(tuple(graphs_labels))
        embedding_size = unique_labels.shape[0]
        print(f'Dimension is {embedding_size}')
        print('Cleaning...')
        unique_labels_to_inds = dict(zip(unique_labels, np.arange(embedding_size)))
        embeddings = [WLEmbeddings.get_embedding(labels, unique_labels_to_inds, embedding_size) for labels in tqdm_notebook(graphs_labels)]
        return embeddings

def WLEmbedsEndToEnd(data, graph_distance, embed_distance, cs=[1, 10, 100, 500],
                     iterations=3, title='', xtitle='Euclidian dist', ytitle='Edit dist'):
    print('Getting embeddings...')
    we = WLEmbeddings()
    E = we.compute_embeddings(data, iterations)
    print('Getting graph distances...')
    edit_distances = get_distances(data[0], data, graph_distance)
    print('Getting embed distances...')
    embed_distances = get_distances(E[0], E, embed_distance)
    plot_hypothesis(edit_distances, embed_distances, title=title, cs=cs, xtitle='Euclidian dist', ytitle='Edit dist')
    test_hypothesis(edit_distances, embed_distances, cs=cs)
    return edit_distances, embed_distances

def GraphletEmbedsEndToEnd(data, graph_distance, embed_distance, cs=[1, 10, 100, 500],
                           k=3, title='', xtitle='Euclidian dist', ytitle='Edit dist'):
    print('Getting embeddings...')
    ge = GraphletEmbeddings()
    E = ge.compute_embeddings(data, k)
    print('Getting graph distances...')
    edit_distances = get_distances(data[0], data, graph_distance)
    print('Getting embed distances...')
    embed_distances = get_distances(E[0], E, embed_distance)
    plot_hypothesis(edit_distances, embed_distances, title=title, cs=cs, xtitle='Euclidian dist', ytitle='Edit dist')
    test_hypothesis(edit_distances, embed_distances, cs=cs)
    return edit_distances, embed_distances
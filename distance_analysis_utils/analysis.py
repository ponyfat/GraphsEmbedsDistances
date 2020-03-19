import networkx as nx
import matplotlib.pyplot as plt
import random
import os
from functools import partial
from tqdm import tqdm_notebook
import numpy as np
import scipy

def get_distances(from_, others, distance_function):
    distances = []
    for other in tqdm_notebook(others):
        distance = distance_function(from_, other)
        distances.append(distance)
    return np.array(distances)

def save_results(graph_dists, emb_dists, emb_name, ds_name, path, emb_sim=None):
    print(f'{path}/{emb_name}_{ds_name}_gdist')
    np.save(f'{path}/{emb_name}_{ds_name}_gdist', graph_dists)
    np.save(f'{path}/{emb_name}_{ds_name}_edist', emb_dists)
    if emb_sim is not None:
        np.save(f'{path}/{emb_name}_{ds_name}_esim', emb_sim)

def plot_one_hypothesis(ax, graph_distances, embed_distances, c=1, bounds='both'):
    ax.plot(graph_distances, label='Graph dist')
    if bounds == 'both' or bounds == 'lower':
        ax.plot(1 / c * embed_distances, label='Embed dist', c='r', alpha=0.5)
    if bounds == 'both' or bounds == 'upper':
        ax.plot(c * embed_distances, label='Embed dist', c='r', alpha=0.5)
    ax.grid()
    ax.legend()

def plot_hypothesis(graph_distances, embed_distances, title = '', cs=[1, 10, 100, 500], xtitle='', ytitle='', bounds='both'):
    fig, axs = plt.subplots(nrows=1, ncols=len(cs), figsize=(18, 5))
    fig.suptitle(f'{title}, c={[float("%.2f"%item) for item in cs]}', fontsize=16)
    for i, c in enumerate(cs):
        axs[i].set_title(f'c = {float("%.2f"%c)}')
        plot_one_hypothesis(axs[i], graph_distances, embed_distances, c=c, bounds=bounds)
        axs[i].set_xlabel('Item')
        axs[i].set_ylabel(ytitle)

def test_hypothesis(graph_distances, embed_distances, cs):
    correct = []
    for c in cs:
        low_bound = 1 / c * embed_distances
        upper_bound = c * embed_distances
        low_errs = (low_bound > graph_distances).sum()
        up_errs = (upper_bound < graph_distances).sum()
        print(f'{c}: upper bound errs - {up_errs}, low bound errs - {low_errs})')
        if low_errs == 0 and up_errs == 0:
            correct.append(c)
    if len(correct) != 0:
        print(f'Hypothesis is correct for c in {[float("%.2f"%item) for item in correct]}')
    else:
        print('No correct c found')

def find_c(graph_distances, embed_distances, c_max, approx=0.01):
    low_bound = 1 / c_max * embed_distances
    upper_bound = c_max * embed_distances
    low_errs = (low_bound > graph_distances).sum()
    up_errs = (upper_bound < graph_distances).sum()
    if low_errs + up_errs > 0:
        print('optimal c can\'t be found, c_max not enough')
        return None
    r = c_max
    l = 0.000000001
    while (r - l > approx):
        m = (r + l) / 2
        low_bound = 1 / m * embed_distances
        upper_bound = m * embed_distances
        low_errs = (low_bound > graph_distances).sum()
        up_errs = (upper_bound < graph_distances).sum()
        if low_errs == 0 and up_errs == 0:
            r = m
        else:
            l = m
    return r

def fastEdgeDistance(G, other_G):
    return float(np.abs(G.number_of_edges() - other_G.number_of_edges()))

def diffSizedClicksDistance(G, other_G):
    return float(np.abs(G.number_of_edges() - other_G.number_of_edges()) + np.abs(G.number_of_nodes() - other_G.number_of_nodes()))


def find_c_growth(graph_distances, embed_distances, c_max, approx=0.01):
    c_growth = []
    for i in range(0, len(graph_distances)):
        c_growth.append(find_c(graph_distances[i], embed_distances[i], c_max, approx))
    return c_growth

def find_c_growth_inside(graph_distances, embed_distances, c_max, approx=0.01):
    c_growth = []
    for i in range(0, graph_distances.shape[0]):
        c_growth.append(find_c(graph_distances[:i], embed_distances[:i], c_max, approx))
    return c_growth

def plot_c_growth(c_growth, ds_title=''):
    plt.figure(figsize=(7, 5))
    plt.plot(c_growth, label='c growth', c='orange')
    plt.ylabel('Optimal c approximation')
    plt.xlabel('Size of dataset')
    plt.title(f'C growth for {ds_title}')
    plt.grid()
    plt.legend()

def plot_embs_sim(emb_sim, ds_title=''):
    plt.figure(figsize=(7, 5))
    plt.plot(emb_sim, label='Dot product of f(0) and f(i)', c='green')
    plt.ylabel('Similiarity')
    plt.xlabel('i')
    plt.title(f'Dot product similiarity {ds_title}')
    plt.grid()
    plt.legend()

def plot_dependencies(graph_distances, embed_distances, title=''):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle(f'Graphs/Embeddings distances dependencies for {title}', fontsize=16)
    axs[0].plot(embed_distances, label='EmbeddingDist/item', c='r')
    axs[0].set_ylabel('Euclidian distance')
    axs[0].set_xlabel('Item')
    axs[0].set_title(f'Embeddings distance/item')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(graph_distances, label='GraphDist/item')
    axs[1].set_ylabel('Edit distance')
    axs[1].set_xlabel('Item')
    axs[1].set_title(f'Graphs distance/item')
    axs[1].grid()
    axs[1].legend()
   
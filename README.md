# Graphs/Embeds Distances

## Generation (in module distances_analysis.utils.generation)
All with prefix ```generate``` generate a list of networkx objects. All functions wil prefix ```create``` create a directory with indexed graphML objects.

1. **Clicks(d)** - contains d graphs, ![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20G_0%2C%20%5Cldots%2C%20G_%7Bd%20-%201%7D%20%5C%7D), where ![equation](https://latex.codecogs.com/gif.latex?G_i) is a click of size ![equation](https://latex.codecogs.com/gif.latex?i) united with ![equation](https://latex.codecogs.com/gif.latex?d%20-%20i) isolated vertices.

2. **RandomEdgesRemoval(d), RER(d)** - 2. **RandomEdgesRemoval(d), RER(d)** - contains ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%20%28d%20-%201%29%7D%7B2%7D) graphs ![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20G_0%2C%20%5Cldots%2C%20G_%7B%5Cfrac%7Bd%20%28d%20-%201%29%7D%7B2%7D%20-%201%7D%5C%7D), ![equation](https://latex.codecogs.com/gif.latex?G_0) is a click of size d, graph  ![equation](https://latex.codecogs.com/gif.latex?G_i) is received by removing an edge from ![equation](https://latex.codecogs.com/gif.latex?G_%7Bi%20-%201%7D) with equal probabilities for each of the edges.


And additional two:

3. **ComplicatedClicks(d)** - contains d graphs, ![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20G_0%2C%20%5Cldots%2C%20G_%7Bd%20-%201%7D%20%5C%7D), where ![equation](https://latex.codecogs.com/gif.latex?G_i) is a click of size ![equation](https://latex.codecogs.com/gif.latex?i)
united with ![equation](https://latex.codecogs.com/gif.latex?d%20-%20i) vertices connected to vertex 0.

4. **RandomMEdges(d, m)** - contains  graphs ![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20G_0%2C%20%5Cldots%2C%20G_%7B%5Cfrac%7Bd%20%28d%20-%201%29%7D%7B2%7D%20-%201%7D%20%5C%7D), ![equation](https://latex.codecogs.com/gif.latex?G_i) is a random graph with ![equation](https://latex.codecogs.com/gif.latex?n) vertices and ![equation](https://latex.codecogs.com/gif.latex?%2C) edges.

## Embeddings (in module distances_analysis.utils.embeddings)
Require list of networx objects as an input.

1. WLEmbeddings
2. GraphletEmbeddings

Usage:

```
G = generateClicks(5)
ge = GraphletEmbeddings()
embeddings = ge.compute_embeddings(G, 3)
```
## Analysis (in module distances_analysis.utils.analysis)
Plotting/computing c instruments.


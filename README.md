# Graphs/Embeds Distances

## Generation (in module)
1. **Clicks(d)** - contains d graphs, ![equation](https://latex.codecogs.com/gif.latex?%5C%7B%20G_0%2C%20%5Cldots%2C%20G_%7Bd%20-%201%7D%20%5C%7D), where ![equation](https://latex.codecogs.com/gif.latex?G_i) is a click of size ![equation](https://latex.codecogs.com/gif.latex?i) united with ![equation](https://latex.codecogs.com/gif.latex?d%20-%20i) isolated vertices.

2. **RandomEdgesRemoval(d), RER(d)** - contains $\frac{d (d - 1)}{2}$ graphs $\{ G_0, \ldots, G_{\frac{d (d - 1)}{2} - 1}\}$, G_0 is a click of size d, graph  $G_i$ is received by removing an edge from $G_{i - 1}$ with equal probabilities for each of the edges.


And additional two:

3. **ComplicatedClicks(d)** - contains d graphs, $\{ G_0, \ldots, G_{d - 1} \}$, where $G_i$ is a click of size $i$
united with $d - i$ vertices connected to vertex 0.

4. **RandomMEdges(d, m)** - contains  graphs $\{ G_0, \ldots, G_{\frac{d (d - 1)}{2} - 1} \}$, $G_i$ is a random graph with $n$ vertices and $m$ edges.

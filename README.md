# Graphs/Embeds Distances

## Generation (in module)
1. **Clicks(d)** - contains d graphs, $\{ G_0, \ldots, G_{d - 1} \}$, where $G_i$ is a click of size $i$ united with $d - i$ isolated vertices.

2. **RandomEdgesRemoval(d), RER(d)** - contains $\frac{d (d - 1)}{2}$ graphs $\{ G_0, \ldots, G_{\frac{d (d - 1)}{2} - 1}\}$, G_0 is a click of size d, graph  $G_i$ is received by removing an edge from $G_{i - 1}$ with equal probabilities for each of the edges.


And additional two:

3. **ComplicatedClicks(d)** - contains d graphs, $\{ G_0, \ldots, G_{d - 1} \}$, where $G_i$ is a click of size $i$
united with $d - i$ vertices connected to vertex 0.

4. **RandomMEdges(d, m)** - contains  graphs $\{ G_0, \ldots, G_{\frac{d (d - 1)}{2} - 1} \}$, $G_i$ is a random graph with $n$ vertices and $m$ edges.

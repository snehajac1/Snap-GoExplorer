# # save as find_pairs.py at project root and run:  python find_pairs.py

# import random
# import networkx as nx

# from src.graph_loader import load_email_graph

# DATA_PATH = "data/email-Eu-core.txt"
# G: nx.Graph = load_email_graph(DATA_PATH)

# nodes = list(G.nodes())

# def score_pair(u, v):
#     """Return (path_length, num_common_neighbors) for a pair."""
#     try:
#         d = nx.shortest_path_length(G, u, v)
#     except nx.NetworkXNoPath:
#         return None
#     commons = list(nx.common_neighbors(G, u, v))
#     return d, len(commons)

# candidates = []

# # sample random pairs to find fun ones
# for _ in range(2000):
#     u, v = random.sample(nodes, 2)
#     res = score_pair(u, v)
#     if res is None:
#         continue
#     d, c = res
#     # tweak this filter if you want shorter/longer paths
#     if d >= 4 or c >= 5:
#         candidates.append((d, c, u, v))

# # sort “most interesting” first: long path, many common neighbors
# candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

# print("Top 15 interesting pairs (path_len, |common_neighbors|, src, dst):")
# for row in candidates[:15]:
#     print(row)


import networkx as nx
from src.graph_loader import load_email_graph

DATA_PATH = "data/email-Eu-core.txt"
G = load_email_graph(DATA_PATH)

results = []

nodes = list(G.nodes())

for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        u, v = nodes[i], nodes[j]

        # skip adjacent nodes (boring)
        if G.has_edge(u, v):
            continue
        
        # find common neighbors
        commons = list(nx.common_neighbors(G, u, v))
        cn = len(commons)

        if cn == 0:
            continue  # skip uninteresting ones

        # find shortest path length
        try:
            dist = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            continue

        results.append((cn, dist, u, v))

# sort by most common neighbors first, then longest path
results.sort(reverse=True)

print("Top 15 most interesting pairs (common_neighbors, path_len, src, dst):")
for r in results[:15]:
    print(r)

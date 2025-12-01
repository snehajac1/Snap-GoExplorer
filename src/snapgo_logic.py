import networkx as nx

# ============================================================
# UNDIRECTED GRAPH HELPERS
# ============================================================

def shortest_path_nodes_undirected(G, src, dst):
    try:
        return nx.shortest_path(G, src, dst)
    except:
        return []

def common_neighbors_undirected(G, u, v):
    if not G.has_node(u) or not G.has_node(v):
        return []
    return list(nx.common_neighbors(G, v, u))

# ============================================================
# DIRECTED GRAPH HELPERS
# ============================================================

def shortest_path_nodes_directed(G, src, dst):
    try:
        return nx.shortest_path(G, src, dst)
    except:
        return []

def common_in_neighbors_directed(G, u, v):
    """Nodes that have outgoing edges to BOTH u and v (w→u and w→v)."""
    if not G.has_node(u) or not G.has_node(v):
        return []
    Nu = set(G.predecessors(u))     # ? → u
    Nv = set(G.predecessors(v))     # ? → v
    return list(Nu & Nv)


# ============================================================
# MATRIX CELL HELPERS (used by BOTH directed + undirected)
# ============================================================

def map_nodes_to_indices(nodes_order):
    return {node: i for i, node in enumerate(nodes_order)}

def path_cells(path_nodes, node_to_idx):
    cells = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        i = node_to_idx[u]
        j = node_to_idx[v]
        cells.append((i, j))
    return cells

def common_neighbor_cells(u, v, commons, node_to_idx):
    cells = []
    ui = node_to_idx[u]
    vi = node_to_idx[v]

    for w in commons:
        wi = node_to_idx[w]
        cells.append((ui, wi))
        cells.append((vi, wi))
    return cells

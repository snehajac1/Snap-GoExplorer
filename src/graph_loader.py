import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def load_email_graph(path: str) -> nx.Graph:
    """
    Load the SNAP email-Eu-core graph as an undirected graph.
    Assumes an edgelist where each line is 'u v'.
    Relabels nodes to consecutive integers 0..n-1.
    """
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def load_directed_graph(path: str) -> nx.DiGraph:
    """
    Load a directed graph (e.g., CollegeMsg) from an edgelist.

    For CollegeMsg, each line is: 'u v t', we only use u and v.
    """
    G = nx.DiGraph()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            v = int(parts[1])
            G.add_edge(u, v)

    # Relabel to 0..n-1 for consistency
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def graph_to_sparse_adj(G, directed: bool = False):
    """
    Convert a (un)directed graph to a SciPy CSR adjacency matrix.

    If directed=False: treat as undirected, store A[i,j] and A[j,i].
    If directed=True: store only A[i,j] for edge u->v.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}

    rows = []
    cols = []

    if directed:
        # only (i, j) for u -> v
        for u, v in G.edges():
            i = index[u]
            j = index[v]
            rows.append(i)
            cols.append(j)
    else:
        # undirected: add both directions
        for u, v in G.edges():
            i = index[u]
            j = index[v]
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A, nodes




# -----------------------------------------------------------
# Load Email-Eu-core (undirected)
# -----------------------------------------------------------
def load_email_graph(path):
    G = nx.read_edgelist(
        path,
        nodetype=int,
        create_using=nx.Graph()
    )
    return G


# -----------------------------------------------------------
# Load CollegeMsg (directed)
# -----------------------------------------------------------
def load_collegemsg_graph(path):
    G = nx.read_edgelist(
        path,
        nodetype=int,
        create_using=nx.DiGraph()
    )
    return G

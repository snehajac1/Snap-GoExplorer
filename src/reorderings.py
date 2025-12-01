import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import eigsh


def degree_ordering(A: csr_matrix) -> np.ndarray:
    """
    Order nodes by descending degree.
    A is the adjacency matrix in CSR format.
    Returns a permutation array of indices 0..n-1.
    """
    degrees = np.array(A.sum(axis=1)).ravel()
    order = np.argsort(-degrees)  # descending
    return order.astype(int)


def spectral_ordering_from_graph(G: nx.Graph) -> np.ndarray:
    """
    Spectral ordering using the Fiedler vector (2nd smallest eigenvector of Laplacian).
    Returns a permutation array of node indices 0..n-1.
    """
    # Sparse Laplacian matrix
    L = nx.laplacian_matrix(G)

    # SciPy changed csr_matrix → csr_array; use .astype(float) for compatibility
    L = L.astype(float)

    # Compute the two smallest eigenvalues/eigenvectors
    vals, vecs = eigsh(L, k=2, which="SM")

    # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
    fiedler_vec = vecs[:, 1]

    # Sort nodes by Fiedler vector
    order = np.argsort(fiedler_vec)
    return order.astype(int)



def rcm_ordering(A: csr_matrix) -> np.ndarray:
    """
    Reverse Cuthill-McKee ordering for bandwidth reduction.
    A is the adjacency matrix.
    Returns a permutation array of indices 0..n-1.
    """
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
    return np.array(perm, dtype=int)


def permute_matrix(A: csr_matrix, order: np.ndarray) -> csr_matrix:
    """
    Permute rows and columns of A according to the given order array.
    order is a permutation of [0..n-1].
    """
    return A[order, :][:, order]

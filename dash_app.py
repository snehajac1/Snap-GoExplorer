# import numpy as np
# import networkx as nx
# import plotly.graph_objects as go
# from dash import Dash, dcc, html, Input, Output, State

# # -------------------------------
# # IMPORT BOTH UNDIRECTED + DIRECTED LOGIC (YOUR FILE)
# # -------------------------------
# from src.snapgo_logic import (
#     shortest_path_nodes_undirected,
#     shortest_path_nodes_directed,
#     common_neighbors_undirected,
#     common_in_neighbors_directed,
# )

# from src.graph_loader import load_email_graph, graph_to_sparse_adj
# from src.reorderings import (
#     degree_ordering,
#     spectral_ordering_from_graph,
#     rcm_ordering,
#     permute_matrix,
# )

# # ============================================================
# # 1) LOAD **UNDIRECTED** DATASET (A)
# # ============================================================
# DATA_PATH_UNDIRECTED = "data/email-Eu-core.txt"

# G_u = load_email_graph(DATA_PATH_UNDIRECTED)
# A_u, base_nodes_u = graph_to_sparse_adj(G_u)
# n_u = A_u.shape[0]

# orig_order_u = np.arange(n_u)
# deg_order_u = degree_ordering(A_u)
# spec_order_u = spectral_ordering_from_graph(G_u)
# rcm_order_u = rcm_ordering(A_u)

# ORDERS_U = {
#     "Original": orig_order_u,
#     "Degree": deg_order_u,
#     "Spectral": spec_order_u,
#     "RCM": rcm_order_u,
# }

# # ============================================================
# # 2) LOAD **DIRECTED** DATASET (B)
# # ============================================================
# DATA_PATH_DIRECTED = "data/CollegeMsg.txt"

# def load_directed_graph(path):
#     G = nx.DiGraph()
#     with open(path, "r") as f:
#         for line in f:
#             if line.strip():
#                 u, v, *_ = line.split()
#                 u, v = int(u), int(v)
#                 G.add_edge(u, v)
#     return G

# G_d = load_directed_graph(DATA_PATH_DIRECTED)
# A_d, base_nodes_d = graph_to_sparse_adj(G_d)   # This gives adjacency, works for directed
# n_d = A_d.shape[0]

# orig_order_d = np.arange(n_d)
# deg_order_d = degree_ordering(A_d)
# spec_order_d = spectral_ordering_from_graph(G_d.to_undirected())
# rcm_order_d = rcm_ordering(A_d)

# ORDERS_D = {
#     "Original": orig_order_d,
#     "Degree": deg_order_d,
#     "Spectral": spec_order_d,
#     "RCM": rcm_order_d,
# }

# # ============================================================
# # 3) DASH UI SELECTOR
# # ============================================================

# DATASET_OPTIONS = [
#     {"label": "Undirected (Email-Eu-core)", "value": "undirected"},
#     {"label": "Directed (CollegeMsg)", "value": "directed"},
# ]

# ORDERING_OPTIONS = [
#     {"label": "Original", "value": "Original"},
#     {"label": "Degree", "value": "Degree"},
#     {"label": "Spectral", "value": "Spectral"},
#     {"label": "RCM", "value": "RCM"},
# ]

# # ============================================================
# # Helper: Select active dataset
# # ============================================================
# def get_dataset_config(mode):
#     if mode == "directed":
#         return G_d, A_d, ORDERS_D, n_d
#     return G_u, A_u, ORDERS_U, n_u

# # ============================================================
# # Figure Builders
# # ============================================================

# def make_matrix_figure(order_name, orders, A_base, path_cells, path_text, cn_cells, cn_text):
#     order = orders[order_name]
#     A_perm = permute_matrix(A_base, order)
#     M = A_perm.toarray().astype(float)

#     z = np.zeros_like(M)
#     z[M > 0] = 1.0

#     fig = go.Figure()

#     fig.add_trace(go.Heatmap(
#         z=z,
#         colorscale=[[0, "white"], [1, "#e1e3eb"]],
#         showscale=False,
#         hoverinfo="skip",
#     ))

#     if path_cells:
#         px = [c + 0.5 for (r, c) in path_cells]
#         py = [r + 0.5 for (r, c) in path_cells]
#         fig.add_trace(go.Scatter(
#             x=px, y=py,
#             mode="lines+markers",
#             line=dict(color="crimson", width=3),
#             marker=dict(size=10, color="crimson"),
#             text=path_text,
#             hovertemplate="%{text}<extra></extra>",
#             name="Shortest Path"
#         ))

#     if cn_cells:
#         cx = [c + 0.5 for (r, c) in cn_cells]
#         cy = [r + 0.5 for (r, c) in cn_cells]
#         fig.add_trace(go.Scatter(
#             x=cx, y=cy,
#             mode="markers",
#             marker=dict(size=8, color="royalblue"),
#             text=cn_text,
#             hovertemplate="%{text}<extra></extra>",
#             name="Common Neighbors"
#         ))

#     fig.update_layout(
#         margin=dict(l=10, r=10, t=50, b=10),
#         xaxis=dict(showticklabels=False, zeroline=False),
#         yaxis=dict(showticklabels=False, scaleanchor="x", scaleratio=1, autorange="reversed"),
#         title=f"Adjacency Matrix — {order_name}",
#     )
#     return fig


# def make_network_figure(G, src, dst, path_nodes, commons):
#     fig = go.Figure()

#     if src is None or dst is None or len(path_nodes) == 0:
#         fig.update_layout(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             title="Local Network View",
#         )
#         return fig

#     nodes = set(path_nodes) | set(commons) | {src, dst}
#     nodes = sorted(nodes)
#     k = len(nodes)

#     angles = np.linspace(0, 2*np.pi, k, endpoint=False)
#     pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(nodes, angles)}

#     edges = []
#     for u, v in zip(path_nodes[:-1], path_nodes[1:]):
#         edges.append((u, v))
#     for c in commons:
#         edges.append((src, c))
#         edges.append((dst, c))

#     edge_x, edge_y = [], []
#     for u, v in edges:
#         x0, y0 = pos[u]
#         x1, y1 = pos[v]
#         edge_x += [x0, x1, None]
#         edge_y += [y0, y1, None]

#     fig.add_trace(go.Scatter(
#         x=edge_x, y=edge_y,
#         mode="lines",
#         name="Edges",
#         line=dict(color="#cccccc", width=1.5),
#         hoverinfo="skip"
#     ))

#     node_x, node_y, labels, colors = [], [], [], []
#     for n in nodes:
#         x, y = pos[n]
#         node_x.append(x)
#         node_y.append(y)
#         labels.append(str(n))

#         if n == src:
#             colors.append("crimson")
#         elif n == dst:
#             colors.append("darkblue")
#         elif n in path_nodes:
#             colors.append("orange")
#         elif n in commons:
#             colors.append("royalblue")
#         else:
#             colors.append("gray")

#     fig.add_trace(go.Scatter(
#         x=node_x, y=node_y,
#         mode="markers+text",
#         name="Nodes",
#         text=labels,
#         textposition="top center",
#         marker=dict(size=12, color=colors),
#     ))

#     fig.update_layout(
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
#         title="Local Network View"
#     )
#     return fig


# # ============================================================
# # DASH APP
# # ============================================================

# app = Dash(__name__)
# server = app.server

# app.layout = html.Div([
#     html.H2("Matrix Snap&Go — Undirected + Directed"),
#     html.Div(style={"display": "flex", "height": "90vh"}, children=[

#         # --------------------------
#         # LEFT PANEL
#         # --------------------------
#         html.Div(style={"width": "250px", "padding": "15px"}, children=[
#             html.Label("Dataset"),
#             dcc.Dropdown(id="dataset-mode", options=DATASET_OPTIONS, value="undirected"),
#             html.Br(),

#             html.Label("Ordering"),
#             dcc.Dropdown(id="ordering-dropdown", options=ORDERING_OPTIONS, value="Original"),
#             html.Br(),

#             html.Label("Source node"),
#             dcc.Input(id="src-input", type="text", value="0", style={"width": "100%"}),
#             html.Br(), html.Br(),

#             html.Label("Target node"),
#             dcc.Input(id="dst-input", type="text", value="10", style={"width": "100%"}),
#             html.Br(), html.Br(),

#             html.Button("SNAP & GO", id="snapgo-button"),
#             html.Br(), html.Br(),

#             html.Div(id="status-text"),
#         ]),

#         # -------------------------------
#         # MATRIX + LOCAL NETWORK VIEW
#         # -------------------------------
#         html.Div(style={"flex": "1", "display": "flex"}, children=[
#             dcc.Graph(id="matrix-view", config={"displayModeBar": False}, style={"flex": "1"}),
#             dcc.Graph(id="network-view", config={"displayModeBar": False}, style={"width": "420px"}),
#         ]),
#     ])
# ])


# # ============================================================
# # CALLBACK
# # ============================================================

# @app.callback(
#     [Output("matrix-view", "figure"),
#      Output("network-view", "figure"),
#      Output("status-text", "children")],
#     Input("snapgo-button", "n_clicks"),
#     State("dataset-mode", "value"),
#     State("ordering-dropdown", "value"),
#     State("src-input", "value"),
#     State("dst-input", "value"),
# )
# def update(n, mode, order_name, src_str, dst_str):
#     # pick dataset (A/B)
#     G, A_base, ORDERS, n_nodes = get_dataset_config(mode)

#     # parse node IDs
#     try:
#         src = int(src_str)
#         dst = int(dst_str)
#     except:
#         return {}, {}, "Invalid node IDs."

#     if src not in G or dst not in G:
#         return {}, {}, f"Node IDs must be inside 0–{n_nodes-1}"

#     # choose path + CN fn based on mode
#     if mode == "undirected":
#         path_nodes = shortest_path_nodes_undirected(G, src, dst)
#         commons = common_neighbors_undirected(G, src, dst)
#     else:
#         path_nodes = shortest_path_nodes_directed(G, src, dst)
#         commons = common_in_neighbors_directed(G, src, dst)

#     order = ORDERS[order_name]
#     node_to_idx = {node: i for i, node in enumerate(order)}

#     # matrix edges
#     path_cells, path_text = [], []
#     if path_nodes:
#         for u, v in zip(path_nodes[:-1], path_nodes[1:]):
#             i, j = node_to_idx[u], node_to_idx[v]
#             r, c = (i, j) if i < j else (j, i)
#             path_cells.append((r, c))
#             path_text.append(f"{u} → {v}")

#     cn_cells, cn_text = [], []
#     for w in commons:
#         for u in (src, dst):
#             i, j = node_to_idx[u], node_to_idx[w]
#             r, c = (i, j) if i < j else (j, i)
#             cn_cells.append((r, c))
#             cn_text.append(f"{u} ↔ {w}")

#     status = (
#         f"Dataset={mode.upper()} | Path length={len(path_nodes)-1 if path_nodes else '∞'} | "
#         f"Common neighbors={len(commons)}"
#     )

#     matrix_fig = make_matrix_figure(order_name, ORDERS, A_base, path_cells, path_text, cn_cells, cn_text)
#     network_fig = make_network_figure(G, src, dst, path_nodes, commons)

#     return matrix_fig, network_fig, status


# # ============================================================
# if __name__ == "__main__":
#     app.run(debug=False)

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State

# -------------------------------
# IMPORT UNDIRECTED + DIRECTED LOGIC
# -------------------------------
from src.snapgo_logic import (
    shortest_path_nodes_undirected,
    shortest_path_nodes_directed,
    common_neighbors_undirected,
    common_in_neighbors_directed,
)

from src.graph_loader import load_email_graph, graph_to_sparse_adj
from src.reorderings import (
    degree_ordering,
    spectral_ordering_from_graph,
    rcm_ordering,
    permute_matrix,
)

# ============================================================
# 1) LOAD **UNDIRECTED** DATASET (Email-Eu-core)
# ============================================================
DATA_PATH_UNDIRECTED = "data/email-Eu-core.txt"

G_u = load_email_graph(DATA_PATH_UNDIRECTED)
A_u, base_nodes_u = graph_to_sparse_adj(G_u)
n_u = A_u.shape[0]

orig_order_u = np.arange(n_u)
deg_order_u = degree_ordering(A_u)
spec_order_u = spectral_ordering_from_graph(G_u)
rcm_order_u = rcm_ordering(A_u)

ORDERS_U = {
    "Original": orig_order_u,
    "Degree": deg_order_u,
    "Spectral": spec_order_u,
    "RCM": rcm_order_u,
}

# ============================================================
# 2) LOAD **DIRECTED** DATASET (CollegeMsg)
# ============================================================
DATA_PATH_DIRECTED = "data/CollegeMsg.txt"


def load_directed_graph(path):
    G = nx.DiGraph()
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                u, v, *_ = line.split()
                u, v = int(u), int(v)
                G.add_edge(u, v)
    return G


G_d = load_directed_graph(DATA_PATH_DIRECTED)
A_d, base_nodes_d = graph_to_sparse_adj(G_d)  # works for DiGraph too
n_d = A_d.shape[0]

orig_order_d = np.arange(n_d)
deg_order_d = degree_ordering(A_d)
spec_order_d = spectral_ordering_from_graph(G_d.to_undirected())
rcm_order_d = rcm_ordering(A_d)

ORDERS_D = {
    "Original": orig_order_d,
    "Degree": deg_order_d,
    "Spectral": spec_order_d,
    "RCM": rcm_order_d,
}

# ============================================================
# 3) DROPDOWN OPTIONS
# ============================================================

DATASET_OPTIONS = [
    {"label": "Undirected (Email-Eu-core)", "value": "undirected"},
    {"label": "Directed (CollegeMsg)", "value": "directed"},
]

ORDERING_OPTIONS = [
    {"label": "Original", "value": "Original"},
    {"label": "Degree", "value": "Degree"},
    {"label": "Spectral", "value": "Spectral"},
    {"label": "RCM", "value": "RCM"},
]


def get_dataset_config(mode):
    """Return (G, A_base, ORDERS, n_nodes) for the given mode."""
    if mode == "directed":
        return G_d, A_d, ORDERS_D, n_d
    return G_u, A_u, ORDERS_U, n_u


# ============================================================
# FIGURE BUILDERS
# ============================================================

def make_matrix_figure(order_name, orders, A_base, path_cells, path_text, cn_cells, cn_text):
    """Adjacency matrix with Snap&Go overlays."""
    order = orders[order_name]
    A_perm = permute_matrix(A_base, order)
    M = A_perm.toarray().astype(float)

    z = np.zeros_like(M)
    z[M > 0] = 1.0

    fig = go.Figure()

    # base matrix
    fig.add_trace(
        go.Heatmap(
            z=z,
            colorscale=[[0, "white"], [1, "#e1e3eb"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    # shortest path
    if path_cells:
        px = [c + 0.5 for (r, c) in path_cells]
        py = [r + 0.5 for (r, c) in path_cells]
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="lines+markers",
                line=dict(color="crimson", width=3),
                marker=dict(size=10, color="crimson"),
                text=path_text,
                hovertemplate="%{text}<extra></extra>",
                name="Shortest path",
            )
        )

    # common neighbors
    if cn_cells:
        cx = [c + 0.5 for (r, c) in cn_cells]
        cy = [r + 0.5 for (r, c) in cn_cells]
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="markers",
                marker=dict(size=8, color="royalblue"),
                text=cn_text,
                hovertemplate="%{text}<extra></extra>",
                name="Common neighbors",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            autorange="reversed",
        ),
        title=f"Adjacency Matrix — {order_name} ordering",
    )
    return fig


def make_empty_matrix_figure():
    """Default matrix when the app first loads (undirected, no overlays)."""
    return make_matrix_figure("Original", ORDERS_U, A_u, [], [], [], [])


def make_network_figure(src, dst, path_nodes, commons):
    """
    Circular local network view with a clear legend:

    - grey edges
    - red:   source
    - navy:  target
    - orange: nodes on the shortest path (besides src/dst)
    - blue:  common neighbors
    """
    fig = go.Figure()

    if src is None or dst is None:
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            title="Local Network View",
        )
        return fig

    nodes = set()
    if path_nodes:
        nodes.update(path_nodes)
    if commons:
        nodes.update(commons)
    nodes.add(src)
    nodes.add(dst)

    if not nodes:
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
            title="Local Network View",
        )
        return fig

    nodes = sorted(nodes)
    k = len(nodes)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

    # --- edges: path edges + src/dst–common neighbor edges
    edge_pairs = set()
    if path_nodes and len(path_nodes) >= 2:
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            edge_pairs.add((u, v))

    for c in commons or []:
        edge_pairs.add((src, c))
        edge_pairs.add((dst, c))

    edge_x, edge_y = [], []
    for u, v in edge_pairs:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#cfd2dc", width=1.5),
            hoverinfo="skip",
            name="Edges",
        )
    )

    # --- nodes, split into categories so legend is clear
    path_set = set(path_nodes or [])
    commons_set = set(commons or [])

    # Source
    sx, sy = pos[src]
    fig.add_trace(
        go.Scatter(
            x=[sx],
            y=[sy],
            mode="markers+text",
            text=[str(src)],
            textposition="top center",
            marker=dict(size=13, color="crimson"),
            name="Source",
        )
    )

    # Target
    tx, ty = pos[dst]
    fig.add_trace(
        go.Scatter(
            x=[tx],
            y=[ty],
            mode="markers+text",
            text=[str(dst)],
            textposition="top center",
            marker=dict(size=13, color="#0d47a1"),
            name="Target",
        )
    )

    # Path nodes (excluding src/dst)
    px, py, ptext = [], [], []
    for node in nodes:
        if node in path_set and node not in {src, dst}:
            x, y = pos[node]
            px.append(x)
            py.append(y)
            ptext.append(str(node))
    if px:
        fig.add_trace(
            go.Scatter(
                x=px,
                y=py,
                mode="markers+text",
                text=ptext,
                textposition="top center",
                marker=dict(size=12, color="#fb8c00"),
                name="Path nodes",
            )
        )

    # Common neighbors
    cx, cy, ctext = [], [], []
    for node in nodes:
        if node in commons_set:
            x, y = pos[node]
            cx.append(x)
            cy.append(y)
            ctext.append(str(node))
    if cx:
        fig.add_trace(
            go.Scatter(
                x=cx,
                y=cy,
                mode="markers+text",
                text=ctext,
                textposition="top center",
                marker=dict(size=11, color="#b619d2"),
                name="Common neighbors",
            )
        )

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        title="Local Network View",
    )
    return fig


def make_empty_network_figure():
    """Default local network view on app load."""
    return make_network_figure(None, None, [], [])


# ============================================================
# DASH APP
# ============================================================

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H2("Matrix Snap&Go — Undirected + Directed"),
        html.Div(
            style={"display": "flex", "height": "90vh"},
            children=[
                # LEFT PANEL
                html.Div(
                    style={"width": "250px", "padding": "15px"},
                    children=[
                        html.Label("Dataset"),
                        dcc.Dropdown(
                            id="dataset-mode",
                            options=DATASET_OPTIONS,
                            value="undirected",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Ordering"),
                        dcc.Dropdown(
                            id="ordering-dropdown",
                            options=ORDERING_OPTIONS,
                            value="Original",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Source node"),
                        dcc.Input(
                            id="src-input",
                            type="text",
                            value="0",
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Label("Target node"),
                        dcc.Input(
                            id="dst-input",
                            type="text",
                            value="10",
                            style={"width": "100%"},
                        ),
                        html.Br(),
                        html.Br(),
                        html.Button("SNAP & GO", id="snapgo-button"),
                        html.Br(),
                        html.Br(),
                        html.Div(id="status-text"),
                    ],
                ),
                # MATRIX + LOCAL NETWORK
                html.Div(
                    style={"flex": "1", "display": "flex"},
                    children=[
                        dcc.Graph(
                            id="matrix-view",
                            figure=make_empty_matrix_figure(),
                            config={"displayModeBar": False},
                            style={"flex": "1"},
                        ),
                        dcc.Graph(
                            id="network-view",
                            figure=make_empty_network_figure(),
                            config={"displayModeBar": False},
                            style={"width": "420px"},
                        ),
                    ],
                ),
            ],
        ),
    ]
)


# ============================================================
# CALLBACK
# ============================================================

@app.callback(
    [
        Output("matrix-view", "figure"),
        Output("network-view", "figure"),
        Output("status-text", "children"),
    ],
    Input("snapgo-button", "n_clicks"),
    State("dataset-mode", "value"),
    State("ordering-dropdown", "value"),
    State("src-input", "value"),
    State("dst-input", "value"),
)
def update(n_clicks, mode, order_name, src_str, dst_str):
    # pick dataset
    G, A_base, ORDERS, n_nodes = get_dataset_config(mode)

    # parse node IDs
    try:
        src = int(str(src_str).strip())
        dst = int(str(dst_str).strip())
    except Exception:
        status = "Source and target node IDs must be integers."
        return (
            make_empty_matrix_figure(),
            make_empty_network_figure(),
            status,
        )

    if src not in G or dst not in G:
        status = f"Node IDs must be in [0, {n_nodes - 1}]."
        return (
            make_empty_matrix_figure(),
            make_empty_network_figure(),
            status,
        )

    # choose graph logic based on mode
    if mode == "undirected":
        path_nodes = shortest_path_nodes_undirected(G, src, dst)
        commons = common_neighbors_undirected(G, src, dst)
    else:
        path_nodes = shortest_path_nodes_directed(G, src, dst)
        commons = common_in_neighbors_directed(G, src, dst)

    order = ORDERS[order_name]
    node_to_idx = {node: i for i, node in enumerate(order)}

    # matrix overlays
    path_cells, path_text = [], []
    if path_nodes:
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            i, j = node_to_idx[u], node_to_idx[v]
            r, c = (i, j) if i < j else (j, i)
            path_cells.append((r, c))
            path_text.append(f"{u} → {v}")

    cn_cells, cn_text = [], []
    for w in commons:
        for u in (src, dst):
            i, j = node_to_idx[u], node_to_idx[w]
            r, c = (i, j) if i < j else (j, i)
            cn_cells.append((r, c))
            cn_text.append(f"{u} ↔ {w}")

    path_len = len(path_nodes) - 1 if path_nodes else "∞"
    status = (
        f"Dataset={mode.upper()} | Path length={path_len} | "
        f"Common neighbors={len(commons)}"
    )

    matrix_fig = make_matrix_figure(
        order_name, ORDERS, A_base, path_cells, path_text, cn_cells, cn_text
    )
    network_fig = make_network_figure(src, dst, path_nodes, commons)

    return matrix_fig, network_fig, status


# ============================================================
if __name__ == "__main__":
    app.run(debug=False)

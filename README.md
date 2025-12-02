# Snap-GoExplorer
COSC 6344 - Visualization Final Project 

This project is a simplified and extended version of the Snap&Go adjacency-matrix visualization technique. It is designed for exploring paths, common neighbors, and structural relationships in large directed and undirected graphs.

The tool provides an interactive interface built with Dash and Plotly, and it uses NetworkX for graph processing. Users can select a dataset, choose a matrix ordering, and specify a source and target node. The system then highlights the shortest path between the nodes and displays any common neighbors. A synchronized local network view is also shown alongside the adjacency matrix.

Two datasets are included: Email-EU-Core (undirected) and CollegeMsg (directed). The app supports four matrix reorderings: Original, Degree, Spectral, and Reverse Cuthill–McKee. Each ordering reveals different structural patterns in the graph.

This project was completed as part of COSC 6344 (Fall 2025). It demonstrates the advantages of adjacency matrices for path exploration and provides additional capabilities beyond the original Snap&Go technique.

To run the project, create a virtual environment, install the dependencies from requirements.txt, and run the main application file. After launching, open the local server address in a browser to interact with the tool.

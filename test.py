import osmnx as ox
import matplotlib.pyplot as plt

# Descargar el grafo de calles de Lima, Perú
lima_graph = ox.graph_from_place("Lima, Peru", network_type="drive")

# Guardar el grafo en un archivo para reutilizarlo
ox.save_graphml(lima_graph, "lima_graph.graphml")

# Visualizar el grafo
ox.plot_graph(lima_graph)

fig, ax = ox.plot_graph(lima_graph, figsize=(10, 10), node_size=5, edge_linewidth=0.5)
plt.show()



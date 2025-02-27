import osmnx as ox
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# Cargar el grafo de Miraflores
miraflores_graph = ox.graph_from_place("Miraflores, Lima, Peru", network_type="drive")

# Cargar JSON de paquetes
with open("paquetes.json", "r") as f:
    paquetes = json.load(f)

# Asignar cada paquete a su nodo más cercano en el grafo
for paquete in paquetes:
    paquete["nodo"] = ox.distance.nearest_nodes(miraflores_graph, X=paquete["lon"], Y=paquete["lat"])

# Obtener los nodos en una matriz
nodos_lista = np.array([paquete["nodo"] for paquete in paquetes]).reshape(-1, 1)

# Aplicar K-Means (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(nodos_lista)

# Aplicar DBSCAN (detección de grupos densos)
dbscan = DBSCAN(eps=0.01, min_samples=2)
dbscan_labels = dbscan.fit_predict(nodos_lista)

# Asignar etiquetas a los paquetes
for i, paquete in enumerate(paquetes):
    paquete["zona_kmeans"] = kmeans_labels[i]
    paquete["zona_dbscan"] = dbscan_labels[i]

# Graficar resultados
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Plot con K-Means
ox.plot_graph(miraflores_graph, ax=axes[0], show=False, close=False)
for i, paquete in enumerate(paquetes):
    x, y = miraflores_graph.nodes[paquete["nodo"]]["x"], miraflores_graph.nodes[paquete["nodo"]]["y"]
    axes[0].scatter(x, y, c=f"C{paquete['zona_kmeans']}", label=f"Zona {paquete['zona_kmeans']}")
axes[0].set_title("Clustering con K-Means")

# Plot con DBSCAN
ox.plot_graph(miraflores_graph, ax=axes[1], show=False, close=False)
for i, paquete in enumerate(paquetes):
    x, y = miraflores_graph.nodes[paquete["nodo"]]["x"], miraflores_graph.nodes[paquete["nodo"]]["y"]
    axes[1].scatter(x, y, c=f"C{paquete['zona_dbscan']}", label=f"Zona {paquete['zona_dbscan']}")
axes[1].set_title("Clustering con DBSCAN")

plt.show()

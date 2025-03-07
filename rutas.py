import osmnx as ox
import os

import geopandas as gpd
import matplotlib.pyplot as plt

import json

import numpy as np
from numpy import radians, sin, cos, arctan2, sqrt

import pandas as pd

from scipy.spatial.distance import pdist, squareform

import folium

from typing import List, Dict

from sklearn.cluster import SpectralClustering, AffinityPropagation, DBSCAN, KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture

from shapely.geometry import Point, Polygon

import time

from folium.plugins import FastMarkerCluster

import math


# ----- Funciones -----

def tsp_distance(cluster):
    coords = cluster['packages'][['lat', 'lon']].values
    if len(coords) < 2:
        return 0
    current = coords[0]
    remaining = list(coords[1:])
    distance = 0
    while remaining:
        next_point = min(remaining, key=lambda p: np.linalg.norm(current - p))
        distance += np.linalg.norm(current - next_point)
        current = next_point
        remaining.remove(next_point)
    # Volver al punto de inicio
    distance += np.linalg.norm(current - coords[0])
    return distance

def obtener_grafo_miraflores(place_name="Miraflores, Lima, Peru", graphml_file="miraflores_network.graphml"):
    # Verificar si el archivo GraphML ya existe
    # TODO: verificar que tan reciente es el archivo (MAX_AGE = 1 mes)
    if os.path.exists(graphml_file):
        print(f"El archivo {graphml_file} ya existe. Cargando el grafo...")
        graph = ox.load_graphml(graphml_file)
    else:
        print(f"Descargando la red vial de {place_name}...")
        # Descargar la red vial
        graph = ox.graph_from_place(place_name, network_type="drive")
        # Guardar el grafo en formato GraphML
        ox.save_graphml(graph, filepath=graphml_file)
        print(f"Grafo guardado en {graphml_file}")

    return graph

def visualizar_grafo(grafo):
    # Visualizar el grafo
    fig, ax = ox.plot_graph(grafo, node_size=5, edge_linewidth=1, show=False, close=False)
    plt.title("Red Vial de Miraflores")
    plt.show()

def mostrar_informacion_del_grafo(graph):
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

    # Mostrar información básica
    print(f"Número de nodos: {len(nodes_gdf)}")
    print(f"Número de aristas: {len(edges_gdf)}")
    print("\nEjemplo de arista (calle):")
    print(edges_gdf[["length", "geometry", "highway", "oneway"]].head(3))

def obtener_paquetes_con_coordenadas_aux(json_files_names):
    paquetes = []
    for json_file_name in json_files_names:
        with open("paquetes/" + json_file_name, "r") as f:
            paquetes.extend(json.load(f))

    df = pd.read_excel("datos_de_pesos_y_guias.xlsx", engine="openpyxl")
    filtered_df = df#[
        #(df["LOCALIDAD"] == "MIRAFLORES - LIMA - LIMA") &
        #(df["FECHA AO"] == "21/02/2025") &
        #(df["FECHA VISITA 1"] == "22/02/2025") &
        #(df["DESTINO"] == "LIM")
    #]
    # ahora solo obtener en este df el "PESO BALANZA", "PESO VOLUMEN" y "GUIA"
    filtered_df = filtered_df[["PESO BALANZA", "PESO VOLUMEN", "GUIA"]]

    paquetes_coordinadas_only = []

    for idx, paquete in enumerate(paquetes):
        coordenadas = paquete["coordenadas"]
        if coordenadas == "0 , 0":
            coordenadas = paquete["coordenada_puerta"]
        lat, lon = map(float, map(str.strip, coordenadas.split(",")))
        id = idx

        # find the paquete["numero_guia"] === filtered_df["GUIA"] and get the "PESO BALANZA" and "PESO VOLUMEN"
        peso_values = filtered_df[filtered_df["GUIA"] == paquete["numero_guia"]]["PESO BALANZA"].values
        # Verificar si hay valores antes de acceder al índice 0
        if len(peso_values) > 0:
            weight = peso_values[0]
        else:
            print("No se encontró el peso del paquete con guia: ", paquete["numero_guia"])
            weight = None  # O algún valor por defecto

        peso_volumen_values = filtered_df[filtered_df["GUIA"] == paquete["numero_guia"]]["PESO VOLUMEN"].values
        if len(peso_volumen_values) > 0:
            volume = (peso_volumen_values[0] * 5000)/1000000
        else:
            print("No se encontró el volumen del paquete con guia: ", paquete["numero_guia"])
            volume = None

        # verify exist if not exist, then don't append
        if weight == None or volume == None:
            continue
        paquetes_coordinadas_only.append({ "id": id, "lat": lat, "lon": lon, "direccion": paquete["dir_calle"],  "weight": weight, "volume": volume, "numero_guia": paquete["numero_guia"] })

    return paquetes_coordinadas_only
 
def obtener_paquetes_con_coordenadas(json_files_names):
    paquetes = []
    for json_file_name in json_files_names:
        with open("paquetes/" + json_file_name, "r") as f:
            paquetes.extend(json.load(f))

    # Leer el Excel y quedarnos solo con las columnas relevantes
    df = pd.read_excel("datos_de_pesos_y_guias.xlsx", engine="openpyxl")
    df = df[["GUIA", "PESO BALANZA", "PESO VOLUMEN"]].dropna()

    # Eliminar duplicados dejando el primer registro por cada GUIA
    df = df.groupby("GUIA").first().reset_index()

    # Convertir en diccionario para acceso rápido
    datos_guia = df.set_index("GUIA").to_dict("index")

    paquetes_coordinadas_only = []

    for idx, paquete in enumerate(paquetes):
        coord = paquete["coordenadas"]
        if coord == "0 , 0":
            coord = paquete["coordenada_puerta"]

        try:
            lat, lon = map(float, map(str.strip, coord.split(",")))
        except ValueError:
            print("Error al procesar coordenadas:", coord)
            continue

        guia = paquete["numero_guia"]
        if guia not in datos_guia:
            print("No se encontró datos para la guia:", guia)
            continue

        weight = datos_guia[guia]["PESO BALANZA"]
        volume = (datos_guia[guia]["PESO VOLUMEN"] * 5000) / 1000000

        paquetes_coordinadas_only.append({
            "id": idx,
            "lat": lat,
            "lon": lon,
            "direccion": paquete["dir_calle"],
            "weight": weight,
            "volume": volume,
            "numero_guia": guia
        })

    return paquetes_coordinadas_only

def assign_to_trucks(clusters, trucks):
    """
    Asigna cada cluster al camión más pequeño que pueda transportarlo.
    """
    trucks_sorted = sorted(trucks, key=lambda x: (-x['max_weight'], -x['max_volume']))
    assignments = []
    remaining_clusters = clusters.copy()

    for truck in trucks_sorted:
        if not remaining_clusters:
            break
        for cluster in remaining_clusters:
            if (cluster['total_weight'] <= truck['max_weight'] and 
                cluster['total_volume'] <= truck['max_volume']):
                assignments.append({
                    'truck_id': truck['id'],
                    'cluster': cluster,
                    'remaining_weight': float(truck['max_weight']) - float(cluster['total_weight']),
                    'remaining_volume': float(truck['max_volume']) - float(cluster['total_volume']),
                    'truck': truck
                })
                remaining_clusters.remove(cluster)
                break

    return assignments, remaining_clusters

def multi_capacity_clustering_AUX(packages, trucks, max_retries=3):
    """
    Crea clusters considerando múltiples camiones con capacidades diferentes.
    Para que se priorize primero llenar el camión más grande, se ordenan los camiones de mayor a menor capacidad.
    Se intenta generar clusters que encajen en el camión más grande disponible, y se dividen si no caben.
    Se repite el proceso varias veces para encontrar la asignación óptima.
    """
    # Ordenar camiones de mayor a menor capacidad
    trucks_sorted = sorted(trucks, key=lambda x: (-x['max_weight'], -x['max_volume']))
    
    # Intentar crear clusters que encajen en el camión más grande disponible
    best_assignment = None
    min_trucks = float('inf')
    
    for _ in range(max_retries):
        clusters = []
        remaining_packages = packages.copy()
        
        # Generar clusters basados en el camión más grande
        largest_truck = trucks_sorted[0]
        k_initial = max(
            int(np.ceil(remaining_packages['weight'].sum() / largest_truck['max_weight'])),
            int(np.ceil(remaining_packages['volume'].sum() / largest_truck['max_volume']))
        )
        
        if k_initial == 0:
            break
            
        kmeans = KMeans(n_clusters=k_initial, n_init=10, random_state=np.random.randint(100))
        remaining_packages['cluster'] = kmeans.fit_predict(remaining_packages[['lat', 'lon']])
        
        # Verificar restricciones y dividir clusters
        for cluster_id in remaining_packages['cluster'].unique():
            cluster = remaining_packages[remaining_packages['cluster'] == cluster_id]
            total_weight = cluster['weight'].sum()
            total_volume = cluster['volume'].sum()
            
            # ¿El cluster cabe en algún camión?
            fits_any_truck = any(
                total_weight <= t['max_weight'] and total_volume <= t['max_volume']
                for t in trucks_sorted
            )
            
            if fits_any_truck:
                clusters.append({
                    'total_weight': total_weight,
                    'total_volume': total_volume,
                    'packages': cluster
                })
            else:
                # Dividir el cluster en dos
                sub_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
                sub_labels = sub_kmeans.fit_predict(cluster[['lat', 'lon']])
                for sub_cluster in [cluster[sub_labels == 0], cluster[sub_labels == 1]]:
                    clusters.append({
                        'total_weight': sub_cluster['weight'].sum(),
                        'total_volume': sub_cluster['volume'].sum(),
                        'packages': sub_cluster
                    })
        
        # Asignar clusters a camiones
        assignments, unassigned = assign_to_trucks(clusters, trucks_sorted)
        
        if len(assignments) < min_trucks and not unassigned:
            min_trucks = len(assignments)
            best_assignment = assignments
    
    return best_assignment

def cluster_cost(cluster: Dict, truck: Dict) -> float:
    """
    Calcula el costo de un cluster basado en la distancia intra-cluster y la capacidad sobrante.
    """
    
    # Distancia intra-cluster
    coords = cluster['packages'][['lat', 'lon']].values
    if len(coords) > 1:
        distances = squareform(pdist(coords, metric='euclidean'))
        intra_cluster_distance = distances.sum() / 2  # Suma de distancias
    else:
        intra_cluster_distance = 0
    
    # Penalización por capacidad sobrante
    weight_penalty = max(0, truck['max_weight'] - cluster['total_weight'])
    volume_penalty = max(0, truck['max_volume'] - cluster['total_volume'])
    
    # Costo total
    cost = intra_cluster_distance + 10 * (weight_penalty + volume_penalty)
    return cost


# --- Funciones de agrupación con restricciones de capacidad ---

def capacitated_dbscan(packages: pd.DataFrame, trucks: List[Dict], n_samples=5, eps=0.01, min_samples=5):
    """
    Agrupación con DBSCAN y restricciones de capacidad.
    - packages: DataFrame con columnas ['lat','lon','weight','volume',...]
    - trucks: lista de camiones con capacidad (usamos trucks[0] como referencia de capacidad para bin_packing_split).
    - n_samples: cuántas veces probamos distintos muestreos/semillas para buscar la mejor solución.
    - eps, min_samples: hiperparámetros de DBSCAN.
    """
    best_clusters = None
    best_cost = float('inf')

    for _ in range(n_samples):
        # Tomar una muestra (por ejemplo, 80%) para inicializar
        sample = packages.sample(frac=0.8, random_state=np.random.randint(100))

        # Ajustar DBSCAN sobre la muestra para una idea inicial
        # (aunque en DBSCAN lo más común es entrenar en todos los datos).
        db_sample = DBSCAN(eps=eps, min_samples=min_samples)
        db_sample.fit(sample[['lat', 'lon']])

        # Ajustar DBSCAN en todos los datos para obtener etiquetas
        db_all = DBSCAN(eps=eps, min_samples=min_samples)
        db_all.fit(packages[['lat','lon']])
        labels = db_all.labels_  # -1 significa ruido

        # Crear clústeres ignorando el ruido (label = -1)
        clusters = []
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # eliminamos ruido

        for cluster_id in unique_labels:
            sub_cluster = packages[labels == cluster_id]
            # Dividir si excede capacidad o si supera 100 coordenadas (bin_packing_split)
            sub_clusters = bin_packing_split(sub_cluster, trucks[0])
            clusters.extend(sub_clusters)

        # Calcular costo (distancia interna + penalización por capacidad)
        cost = sum([cluster_cost(c, trucks[0]) for c in clusters])

        # Guardar la mejor partición
        if cost < best_cost:
            best_clusters = clusters
            best_cost = cost

    return best_clusters

def capacitated_spectral(packages: pd.DataFrame, trucks: List[Dict], n_samples=5, n_clusters=3):
    """
    Agrupa los paquetes usando Spectral Clustering y respeta las restricciones de capacidad.
    - packages: DataFrame con columnas ['lat','lon','weight','volume',...]
    - trucks: lista de camiones; se usa trucks[0] como referencia para dividir con bin_packing_split.
    - n_samples: cantidad de veces que se prueba la agrupación para buscar la mejor solución.
    - n_clusters: número de clústeres que queremos obtener.
    """
    best_clusters = None
    best_cost = float('inf')
    
    for _ in range(n_samples):
        # Tomamos una muestra (opcionalmente) para variar el random_state
        sample = packages.sample(frac=0.8, random_state=np.random.randint(100))
        
        # Aplicamos Spectral Clustering a todos los datos
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            random_state=np.random.randint(100),
            affinity='nearest_neighbors'  # o 'rbf', según convenga
        )
        labels = spectral.fit_predict(packages[['lat', 'lon']])
        
        clusters = []
        for cluster_id in range(n_clusters):
            sub_cluster = packages[labels == cluster_id]
            # Dividimos si excede las restricciones de capacidad o el límite de 100 coordenadas
            sub_clusters = bin_packing_split(sub_cluster, trucks[0])
            clusters.extend(sub_clusters)
        
        # Calcular el costo total (distancia interna + penalización de capacidad)
        cost = sum([cluster_cost(c, trucks[0]) for c in clusters])
        
        if cost < best_cost:
            best_cost = cost
            best_clusters = clusters
            
    return best_clusters

def capacitated_clara_gm(packages: pd.DataFrame, trucks: List[Dict], n_samples=5, n_clusters=3):
    """
    Versión adaptada de CLARA usando Gaussian Mixture (en lugar de KMeans),
    con restricciones de capacidad.
    """
    best_clusters = None
    best_cost = float('inf')
    
    for _ in range(n_samples):
        # Tomar muestra aleatoria de los paquetes (ej. 80%)
        sample = packages.sample(frac=0.8, random_state=np.random.randint(100))
        
        # Ajustar Gaussian Mixture con la muestra
        gm = GaussianMixture(n_components=n_clusters, random_state=np.random.randint(100))
        gm.fit(sample[['lat', 'lon']])
        
        # Predecir etiquetas para TODOS los paquetes
        all_labels = gm.predict(packages[['lat', 'lon']])
        
        # Generar clústeres
        clusters = []
        for cluster_id in range(n_clusters):
            sub_cluster = packages[all_labels == cluster_id]
            # Dividir si excede capacidad (similar a bin_packing_split)
            sub_clusters = bin_packing_split(sub_cluster, trucks[0])
            clusters.extend(sub_clusters)
        
        # Calcular “costo” (distancia interna + penalización por capacidad)
        cost = sum([cluster_cost(c, trucks[0]) for c in clusters])
        
        # Ver si es la mejor solución
        if cost < best_cost:
            best_clusters = clusters
            best_cost = cost
    
    return best_clusters

def capacitated_clara(packages: pd.DataFrame, trucks: List[Dict], n_samples=5, n_clusters=3):
    """
    Versión adaptada de CLARA para clustering con restricciones de capacidad.
    """
    best_clusters = None
    best_cost = float('inf')
    
    for _ in range(n_samples):
        # Muestra aleatoria del 80% de los datos
        sample = packages.sample(frac=0.7)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(sample[['lat', 'lon']])
        
        # Asignar todos los puntos a los clusters
        all_labels = kmeans.predict(packages[['lat', 'lon']])
        clusters = []
        
        for cluster_id in range(n_clusters):
            cluster = packages[all_labels == cluster_id]
            # Dividir cluster si excede capacidades
            sub_clusters = bin_packing_split(cluster, trucks[0])
            clusters.extend(sub_clusters)
        
        # Calcular costo (distancia intra-cluster + penalización por capacidad sobrante)
        cost = sum([cluster_cost(c, trucks[0]) for c in clusters])
        if cost < best_cost:
            best_clusters = clusters
            best_cost = cost
    
    return best_clusters

def capacitated_affinity_propagation(packages: pd.DataFrame, trucks: List[Dict],
                                      damping=0.9, max_iter=200, convergence_iter=15):
    """
    Agrupa paquetes usando Affinity Propagation.
    """
    affinity = AffinityPropagation(damping=damping, max_iter=max_iter,
                                    convergence_iter=convergence_iter, random_state=42)
    labels = affinity.fit_predict(packages[['lat', 'lon']])
    
    clusters = []
    for cluster_id in set(labels):
        sub_cluster = packages[labels == cluster_id]
        sub_clusters = bin_packing_split(sub_cluster, trucks[0])
        clusters.extend(sub_clusters)
        
    return clusters

def capacitated_meanshift(packages: pd.DataFrame, trucks: List[Dict],
                          quantile=0.2, n_samples=500):
    """
    Agrupa paquetes usando MeanShift.
    """
    bandwidth = estimate_bandwidth(packages[['lat', 'lon']], quantile=quantile, n_samples=n_samples)
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = meanshift.fit_predict(packages[['lat', 'lon']])
    
    clusters = []
    for cluster_id in set(labels):
        sub_cluster = packages[labels == cluster_id]
        sub_clusters = bin_packing_split(sub_cluster, trucks[0])
        clusters.extend(sub_clusters)
        
    return clusters

def capacitated_ward(packages: pd.DataFrame, trucks: List[Dict], n_clusters=3):
    """
    Agrupa paquetes usando Agglomerative Clustering con enlace 'ward'.
    """
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = ward.fit_predict(packages[['lat', 'lon']])
    
    clusters = []
    for cluster_id in range(n_clusters):
        sub_cluster = packages[labels == cluster_id]
        sub_clusters = bin_packing_split(sub_cluster, trucks[0])
        clusters.extend(sub_clusters)
        
    return clusters

def capacitated_agglomerative(packages: pd.DataFrame, trucks: List[Dict],
                              n_clusters=3, linkage='average'):
    """
    Agrupa paquetes usando Agglomerative Clustering con enlace especificado (e.g., 'average').
    """
    agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agglo.fit_predict(packages[['lat', 'lon']])
    
    clusters = []
    for cluster_id in range(n_clusters):
        sub_cluster = packages[labels == cluster_id]
        sub_clusters = bin_packing_split(sub_cluster, trucks[0])
        clusters.extend(sub_clusters)
        
    return clusters

def capacitated_birch(packages: pd.DataFrame, trucks: List[Dict],
                      threshold=0.5, n_clusters=3):
    """
    Agrupa paquetes usando Birch.
    """
    birch = Birch(threshold=threshold, n_clusters=n_clusters)
    labels = birch.fit_predict(packages[['lat', 'lon']])
    
    clusters = []
    for cluster_id in range(n_clusters):
        sub_cluster = packages[labels == cluster_id]
        sub_clusters = bin_packing_split(sub_cluster, trucks[0])
        clusters.extend(sub_clusters)
        
    return clusters

# --- FIN DE Funciones de agrupación con restricciones de capacidad ---

def bin_packing_split_1(cluster: pd.DataFrame, truck: Dict):
    """
    Divide un cluster en subclusters de máximo 100 coordenadas únicas (lat, lon),
    respetando las restricciones de peso y volumen del camión.
    """
    clusters = []
    
    # Agrupar paquetes por coordenadas únicas
    coord_groups = cluster.groupby(['lat', 'lon'])
    
    current_cluster = {
        'total_weight': 0,
        'total_volume': 0,
        'packages': pd.DataFrame(columns=cluster.columns),
        'unique_coords': set()
    }
    
    for (lat, lon), packages in coord_groups:
        # Si añadir esta coordenada excede las 100 únicas, se inicia un nuevo clúster
        if len(current_cluster['unique_coords']) >= 100:
            clusters.append(current_cluster)
            current_cluster = {
                'total_weight': 0,
                'total_volume': 0,
                'packages': pd.DataFrame(columns=cluster.columns),
                'unique_coords': set()
            }
        
        # Agregar paquetes de esta coordenada al clúster actual
        if current_cluster['packages'].empty:
            current_cluster['packages'] = packages
        else:
            current_cluster['packages'] = pd.concat([current_cluster['packages'], packages])

        current_cluster['total_weight'] += packages['weight'].sum()
        current_cluster['total_volume'] += packages['volume'].sum()
        current_cluster['unique_coords'].add((lat, lon))
    
    # Agregar el último clúster si contiene paquetes
    if not current_cluster['packages'].empty:
        clusters.append(current_cluster)
    
    return clusters

def bin_packing_split_2(cluster: pd.DataFrame, truck: dict, distance_threshold=5):
    """
    Divide un cluster en subclusters de máximo 100 coordenadas únicas, respetando
    las restricciones de peso y volumen del camión, e incorporando un criterio espacial.
    """
    clusters = []
    
    # Agrupar paquetes por coordenadas únicas
    coord_groups = cluster.groupby(['lat', 'lon'])
    
    current_cluster = {
        'total_weight': 0,
        'total_volume': 0,
        'packages': pd.DataFrame(columns=cluster.columns),
        'unique_coords': set()
    }
    
    for (lat, lon), packages in coord_groups:
        # Calcular el peso y volumen que se sumarían
        new_weight = current_cluster['total_weight'] + packages['weight'].sum()
        new_volume = current_cluster['total_volume'] + packages['volume'].sum()
        
        # Determinar si la nueva coordenada está muy lejos del centroide actual
        add_new_cluster = False
        if current_cluster['unique_coords']:
            coords = list(current_cluster['unique_coords'])
            centroid_lat = sum(coord[0] for coord in coords) / len(coords)
            centroid_lon = sum(coord[1] for coord in coords) / len(coords)
            # Aproximación Euclídea (1 grado ≈ 111 km)
            distance = ((centroid_lat - lat)**2 + (centroid_lon - lon)**2)**0.5 * 111
            if distance > distance_threshold:
                add_new_cluster = True

        # Si se exceden las restricciones o el criterio espacial se cumple, se inicia un nuevo clúster
        if (new_weight > truck['max_weight'] or 
            new_volume > truck['max_volume'] or 
            len(current_cluster['unique_coords']) >= 100 or 
            add_new_cluster):
            clusters.append(current_cluster)
            current_cluster = {
                'total_weight': 0,
                'total_volume': 0,
                'packages': pd.DataFrame(columns=cluster.columns),
                'unique_coords': set()
            }
        
        # Agregar paquetes de esta coordenada
        if current_cluster['packages'].empty:
            current_cluster['packages'] = packages
        else:
            current_cluster['packages'] = pd.concat([current_cluster['packages'], packages])
        current_cluster['total_weight'] += packages['weight'].sum()
        current_cluster['total_volume'] += packages['volume'].sum()
        current_cluster['unique_coords'].add((lat, lon))
    
    # Agregar el último clúster si contiene paquetes
    if not current_cluster['packages'].empty:
        clusters.append(current_cluster)
    
    # Fusión post-proceso de clústers geográficamente cercanos
    merged_clusters = merge_clusters(clusters, truck, distance_threshold)
    return merged_clusters

def merge_clusters(clusters, truck: dict, distance_threshold=5):
    """
    Fusiona clústers que están geográficamente cerca y cuya combinación respeta
    las restricciones de peso y volumen del camión.
    """
    merged = []
    used = [False] * len(clusters)
    
    for i in range(len(clusters)):
        if used[i]:
            continue
        base = clusters[i]
        base_coords = list(base['unique_coords'])
        centroid_lat = sum(coord[0] for coord in base_coords) / len(base_coords)
        centroid_lon = sum(coord[1] for coord in base_coords) / len(base_coords)
        
        merged_cluster = {
            'total_weight': base['total_weight'],
            'total_volume': base['total_volume'],
            'packages': base['packages'],
            'unique_coords': set(base['unique_coords'])
        }
        used[i] = True
        
        for j in range(i+1, len(clusters)):
            if used[j]:
                continue
            other = clusters[j]
            other_coords = list(other['unique_coords'])
            other_centroid_lat = sum(coord[0] for coord in other_coords) / len(other_coords)
            other_centroid_lon = sum(coord[1] for coord in other_coords) / len(other_coords)
            
            # Aproximación Euclídea convertida a km
            distance = (((centroid_lat - other_centroid_lat)**2 + (centroid_lon - other_centroid_lon)**2)**0.5) * 111
            if distance <= distance_threshold:
                # Verificar que al fusionar se cumplan las restricciones
                if (merged_cluster['total_weight'] + other['total_weight'] <= truck['max_weight'] and 
                    merged_cluster['total_volume'] + other['total_volume'] <= truck['max_volume']):
                    merged_cluster['total_weight'] += other['total_weight']
                    merged_cluster['total_volume'] += other['total_volume']
                    merged_cluster['packages'] = pd.concat([merged_cluster['packages'], other['packages']])
                    merged_cluster['unique_coords'].update(other['unique_coords'])
                    used[j] = True
        merged.append(merged_cluster)
    return merged

def bin_packing_split(cluster: pd.DataFrame, truck: Dict, max_coords=100, max_distance_km=2.0):
    """
    Divide un cluster en subclusters respetando:
    1. Máximo de coordenadas únicas por cluster
    2. Proximidad espacial (grupos cercanos juntos)
    3. Restricciones de peso y volumen del camión
    """
    # Si el cluster está vacío o ya cumple con las restricciones, devolverlo directamente
    if cluster.empty or (
        len(cluster.groupby(['lat', 'lon'])) <= max_coords and
        cluster['weight'].sum() <= truck['max_weight'] and
        cluster['volume'].sum() <= truck['max_volume']
    ):
        return [{
            'total_weight': cluster['weight'].sum(),
            'total_volume': cluster['volume'].sum(),
            'packages': cluster,
            'unique_coords': set(zip(cluster['lat'], cluster['lon']))
        }]
    
    # Paso 1: Calcular el centro geográfico del cluster
    center_lat = cluster['lat'].mean()
    center_lon = cluster['lon'].mean()
    
    # Paso 2: Calcular la distancia de cada coordenada al centro
    coord_groups = []
    for (lat, lon), group in cluster.groupby(['lat', 'lon']):
        # Calcular distancia al centro usando la fórmula de Haversine
        distance = haversine_distance(center_lat, center_lon, lat, lon)
        coord_groups.append({
            'lat': lat,
            'lon': lon,
            'distance': distance,
            'packages': group,
            'weight': group['weight'].sum(),
            'volume': group['volume'].sum()
        })
    
    # Paso 3: Ordenar las coordenadas por distancia al centro (más cercanas primero)
    coord_groups.sort(key=lambda x: x['distance'])
    
    # Paso 4: Formar clusters con restricciones espaciales y de capacidad
    clusters = []
    current_cluster = {
        'total_weight': 0,
        'total_volume': 0,
        'packages': pd.DataFrame(columns=cluster.columns),
        'unique_coords': set(),
        'center_lat': center_lat,
        'center_lon': center_lon
    }
    
    for coord_group in coord_groups:
        # Verificar si agregar este grupo excede las capacidades o el límite de coordenadas
        if (len(current_cluster['unique_coords']) + 1 > max_coords or
            current_cluster['total_weight'] + coord_group['weight'] > truck['max_weight'] or
            current_cluster['total_volume'] + coord_group['volume'] > truck['max_volume']):
            
            # Si el cluster actual no está vacío, guardarlo
            if not current_cluster['packages'].empty:
                # Recalcular el centro del cluster actual
                if len(current_cluster['unique_coords']) > 0:
                    current_cluster['center_lat'] = current_cluster['packages']['lat'].mean()
                    current_cluster['center_lon'] = current_cluster['packages']['lon'].mean()
                
                clusters.append({
                    'total_weight': current_cluster['total_weight'],
                    'total_volume': current_cluster['total_volume'],
                    'packages': current_cluster['packages'],
                    'unique_coords': current_cluster['unique_coords']
                })
            
            # Iniciar un nuevo cluster
            current_cluster = {
                'total_weight': 0,
                'total_volume': 0,
                'packages': pd.DataFrame(columns=cluster.columns),
                'unique_coords': set(),
                'center_lat': center_lat,
                'center_lon': center_lon
            }
        
        # Verificar la distancia máxima para mantener la cohesión espacial
        # Si el cluster no está vacío, verificar que la nueva coordenada no esté demasiado lejos
        if not current_cluster['packages'].empty:
            new_center_lat = (current_cluster['packages']['lat'].sum() + coord_group['packages']['lat'].sum()) / (
                len(current_cluster['packages']) + len(coord_group['packages'])
            )
            new_center_lon = (current_cluster['packages']['lon'].sum() + coord_group['packages']['lon'].sum()) / (
                len(current_cluster['packages']) + len(coord_group['packages'])
            )
            
            # Calcular la distancia máxima desde cualquier punto al nuevo centro
            max_dist = 0
            for lat, lon in list(current_cluster['unique_coords']) + [(coord_group['lat'], coord_group['lon'])]:
                dist = haversine_distance(new_center_lat, new_center_lon, lat, lon)
                max_dist = max(max_dist, dist)
            
            # Si la distancia máxima es mayor que el umbral, iniciar un nuevo cluster
            if max_dist > max_distance_km and not current_cluster['packages'].empty:
                clusters.append({
                    'total_weight': current_cluster['total_weight'],
                    'total_volume': current_cluster['total_volume'],
                    'packages': current_cluster['packages'],
                    'unique_coords': current_cluster['unique_coords']
                })
                
                # Iniciar un nuevo cluster con este grupo
                current_cluster = {
                    'total_weight': 0,
                    'total_volume': 0,
                    'packages': pd.DataFrame(columns=cluster.columns),
                    'unique_coords': set(),
                    'center_lat': center_lat,
                    'center_lon': center_lon
                }
        
        # Agregar este grupo al cluster actual
        if current_cluster['packages'].empty:
            current_cluster['packages'] = coord_group['packages']
        else:
            current_cluster['packages'] = pd.concat([current_cluster['packages'], coord_group['packages']])
        
        current_cluster['total_weight'] += coord_group['weight']
        current_cluster['total_volume'] += coord_group['volume']
        current_cluster['unique_coords'].add((coord_group['lat'], coord_group['lon']))
    
    # Agregar el último cluster si contiene paquetes
    if not current_cluster['packages'].empty:
        clusters.append({
            'total_weight': current_cluster['total_weight'],
            'total_volume': current_cluster['total_volume'],
            'packages': current_cluster['packages'],
            'unique_coords': current_cluster['unique_coords']
        })
    
    return clusters

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * arctan2(sqrt(a), sqrt(1-a))

def optimize_residual_space(assignments: List[Dict], trucks: List[Dict]):
    """
    Combina clusters pequeños para llenar espacios residuales, asegurando que no superen las 100 coordenadas únicas.
    """
    for truck in reversed(trucks):  # Empezar con camiones grandes
        for assignment in assignments:
            if assignment['truck_id'] == truck['id']:
                remaining_weight = assignment['remaining_weight']
                remaining_volume = assignment['remaining_volume']
                
                # Buscar clusters pequeños que quepan en el espacio residual
                for candidate in assignments:
                    if (candidate['truck_id'] != truck['id'] and 
                        candidate['cluster']['total_weight'] <= remaining_weight and 
                        candidate['cluster']['total_volume'] <= remaining_volume):

                        # Verificar si la fusión excederá las 100 coordenadas únicas
                        merged_coords = set(assignment['cluster']['packages'][['lat', 'lon']].apply(tuple, axis=1))
                        candidate_coords = set(candidate['cluster']['packages'][['lat', 'lon']].apply(tuple, axis=1))

                        if len(merged_coords | candidate_coords) > 100:
                            continue  # Si excede, no fusionar este candidato

                        # Fusionar clusters
                        assignment['cluster']['packages'] = pd.concat([
                            assignment['cluster']['packages'],
                            candidate['cluster']['packages']
                        ])
                        assignment['cluster']['total_weight'] += candidate['cluster']['total_weight']
                        assignment['remaining_weight'] -= candidate['cluster']['total_weight']
                        assignment['cluster']['total_volume'] += candidate['cluster']['total_volume']
                        assignment['remaining_volume'] -= candidate['cluster']['total_volume']
                        assignments.remove(candidate)
                        break  # Solo fusionar un clúster por iteración
                        
    return assignments

def limitar_paquetes_cluster(clusters):
    for cluster in clusters:
        # Filtrar paquetes únicos según lat y lon
        paquetes_unicos = cluster['packages'].drop_duplicates(subset=['lat', 'lon'])
        # Si hay más de 100, se seleccionan 100 al azar
        if len(paquetes_unicos) > 100:
            paquetes_unicos = paquetes_unicos.sample(n=100, random_state=42)
        # Actualizar el cluster con los paquetes limitados y recalcular totales
        cluster['packages'] = paquetes_unicos
        cluster['total_weight'] = paquetes_unicos['weight'].sum()
        cluster['total_volume'] = paquetes_unicos['volume'].sum()
    return clusters

# --- Modificar la función principal ---
def multi_capacity_clustering(packages, trucks, max_retries=5):
    trucks_sorted = sorted(trucks, key=lambda x: (-x['max_weight'], -x['max_volume']))
    best_assignment = None
    min_trucks = float('inf')
    
    for _ in range(max_retries):
        # Usar CLARA en lugar de K-means puro
        clusters = capacitated_clara(packages, trucks, n_samples=3, n_clusters=len(trucks))
        #clusters = capacitated_clara_gm(packages, trucks, n_samples=5, n_clusters=len(trucks))
        # DESCARTADO clusters = capacitated_dbscan(packages, trucks, n_samples=5, eps=0.01, min_samples=5)
        #clusters = capacitated_spectral(packages, trucks, n_samples=5, n_clusters=len(trucks))
        #clusters = capacitated_meanshift(packages, trucks, quantile=0.2, n_samples=500)
        # DESCARTADO clusters = capacitated_ward(packages, trucks, n_clusters=len(trucks))
        # DESCARTADO clusters = capacitated_agglomerative(packages, trucks, n_clusters=len(trucks), linkage='average')
        # DESCARTADO clusters = capacitated_affinity_propagation(packages, trucks, damping=0.9, max_iter=200, convergence_iter=15)


        # Asignar a camiones
        assignments, unassigned = assign_to_trucks(clusters, trucks_sorted)
        
        # Optimizar espacios residuales
        assignments = optimize_residual_space(assignments, trucks_sorted)
        
        if len(assignments) < min_trucks and not unassigned:
            min_trucks = len(assignments)
            best_assignment = assignments
    
    return best_assignment

def plot_assignments_on_map_aux(assignments):
    # Crear un mapa centrado en Miraflores, Lima
    map_center = [-12.111, -77.031]  # Coordenadas de Miraflores
    m = folium.Map(location=map_center, zoom_start=14)

    points = []  # Lista de coordenadas para FastMarkerCluster

    # Colores para diferenciar camiones
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
                'lightgreen', 'gray', 'black', 'lightgray', 'white']

    # Añadir marcadores para cada camión y sus paquetes
    for i, assign in enumerate(assignments):
        truck_id = i #assign['truck_id']
        cluster = assign['cluster']
        color = colors[i % len(colors)]  # Asignar un color único por camión

        #print(type(cluster['packages']))
        #print(cluster['packages'].head())  # Muestra las primeras filas

        if not isinstance(cluster['packages'], pd.DataFrame):
        #    print("Error: 'packages' no es un DataFrame:", type(cluster['packages']))
            continue  # Evita que el código falle
        if 'lat' not in cluster['packages'].columns or 'lon' not in cluster['packages'].columns:
        #    print("Error: Faltan columnas en 'packages':", cluster['packages'].columns)
            continue  # Evita que el código falle


        # Añadir marcador para el camión (punto de partida)
        folium.Marker(
            location=[cluster['packages']['lat'].mean(), cluster['packages']['lon'].mean()],
            popup=f"Camion: {i + 1}, Paquetes: {len(cluster['packages'])}",
            icon=folium.Icon(color=color, icon='truck', prefix='fa')
        ).add_to(m)

        points.extend([[row['lat'], row['lon']] for _, row in cluster['packages'].iterrows()])

        # Añadir marcadores para los paquetes del camión
        #for _, row in cluster['packages'].iterrows():
            #folium.Marker(
            #    location=[row['lat'], row['lon']],
            #    popup=f"Paquete: {row['weight']} kg, {row['volume']} ton, Camion: {i + 1}",#{truck_id}",
            #    icon=folium.Icon(color=color, icon='box', prefix='fa')
            #).add_to(m)


    FastMarkerCluster(points).add_to(m)  # Renderizar con FastMarkerCluster

    return m

def plot_assignments_on_map(assignments):
    # Crear un mapa centrado en Miraflores, Lima
    map_center = [-12.111, -77.031]
    m = folium.Map(location=map_center, zoom_start=14)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'lightgray', 'white']

    for i, assign in enumerate(assignments):
        truck_id = i
        cluster = assign['cluster']
        color = colors[i % len(colors)]

        if not isinstance(cluster['packages'], pd.DataFrame) or 'lat' not in cluster['packages'].columns or 'lon' not in cluster['packages'].columns:
            continue  # Evita errores

        first_package = cluster['packages'].iloc[0]  # Primer paquete
        truck_lat, truck_lon = first_package['lat'], first_package['lon']

        # Crear un FeatureGroup para el camión y sus paquetes
        truck_group = folium.FeatureGroup(name=f"Camión {i + 1}")

        folium.Marker(
            location=[truck_lat, truck_lon],
            popup=f"Camión: {i + 1}, Paquetes: {len(cluster['packages'])}",
            icon=folium.Icon(color=color, icon='truck', prefix='fa')
        ).add_to(truck_group)


        # Añadir los paquetes dentro del grupo del camión
        for _, row in cluster['packages'].iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"Paquete: {row.get('weight', 'N/A')} kg, {row.get('volume', 'N/A')} ton, Camión: {i + 1}"
            ).add_to(truck_group)

        # Agregar el grupo al mapa
        truck_group.add_to(m)

    # Agregar control de capas para activar/desactivar camiones y paquetes
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def asignaciones_de_paquetes_a_grupos(packages, trucks):
    # Generar asignación óptima
    assignments = multi_capacity_clustering(packages, trucks)
    
    print("")
    # Mostrar resultados
    if assignments:
        print("Asignación de camiones:")
        for i, assign in enumerate(assignments):
            print(f"\nCamión {i + 1}: Peso U: {round(assign['cluster']['total_weight'], 2)} kg, Volumen U: {round(assign['cluster']['total_volume'], 2)} m³")
            print(f"Total de paquetes enviados: {len(assign['cluster']['packages'])}")
            print(f"Capacidad Restante: {round(assign['remaining_weight'], 2)} kg, {round(assign['remaining_volume'], 2)} m³")
        print("")

        total_weight_used = sum(a['cluster']['total_weight'] for a in assignments)
        total_weight_desperdiciado = sum(a['remaining_weight'] for a in assignments)
        total_volume_used = sum(a['cluster']['total_volume'] for a in assignments)
        total_volume_desperdiciado = sum(a['remaining_volume'] for a in assignments)

        print("")
        print(f"Total peso usado: {round(total_weight_used, 2)} kg")# ({
            #total_weight_used / (total_weight_used + total_weight_desperdiciado) * 100:.2f
            #}%) vs sin usar: {total_weight_desperdiciado} kg ({total_weight_desperdiciado / (total_weight_used + total_weight_desperdiciado) * 100:.2f}%)")

        print(f"Total volumen usado: {round(total_volume_used, 2)} m³")# ({
            #total_volume_used  / (total_volume_used + total_volume_desperdiciado) * 100:.2f
            #}%) vs sin usar: {total_volume_desperdiciado} m³ ({total_volume_desperdiciado / (total_volume_used + total_volume_desperdiciado) * 100:.2f}%)")
        print(f"Total camiones usados: {len(assignments)}")
        print(f"Promedio paquetes por camión: {len(packages) / len(assignments):.2f}")

        #print el total de paquetes en los camiones
        total_paquetes = 0
        for assign in assignments:
            total_paquetes += len(assign['cluster']['packages'])
        print(f"Total paquetes en los camiones: {total_paquetes}")
        print("")


        # Visualizar en el mapa
        map_assignments = plot_assignments_on_map(assignments)
        map_assignments.save("mapa_asignaciones.html")  # Guardar el mapa en un archivo HTML
        print("")
        print("Para ver el mapa de asignaciones a grupos, haga click en el siguiente enlace:")
        print("file://" + os.path.abspath("mapa_asignaciones.html"))
        return True
    else:
        print("No se encontró una asignación válida.")
        return False

def asignaciones_de_paquetes_a_locaciones(packages, puntos_files_names):
    # Cargar los puntos de Miraflores desde el JSON
    puntos_miraflores = []
    for puntos_file_name in puntos_files_names:
        with open("puntos/" + puntos_file_name, "r") as f:
            puntos_miraflores.extend(json.load(f))

    assignments = []

    for punto in puntos_miraflores:
        puntos = [list(map(float, pt["map_point"].split(","))) for pt in punto['puntos']]
        poligono = Polygon(puntos)

        punto['paquetes'] = []  # Agregar la clave 'paquetes' al punto

        for _, paquete in packages.iterrows():
            try:
                coordenada = (float(paquete['lat']), float(paquete['lon']))
                punto_geom = Point(coordenada)

                if poligono.contains(punto_geom):
                    punto['paquetes'].append(paquete.to_dict())  # Convertir a diccionario

            except (KeyError, ValueError) as e:
                print(f"⚠️ Error con paquete: {paquete}, {e}")

        assignments.append({
            'truck_id': punto.get('zona_codigo', 'desconocido'),
            'cluster': {
                "packages": pd.DataFrame(punto['paquetes']) 
            }
        })

    map_assignments = plot_assignments_on_map(assignments)
    map_assignments.save("mapa_asignaciones_loc.html")  # Guardar el mapa en un archivo HTML
    print("")
    print("Para ver el mapa de asignaciones a locaciones, haga click en el siguiente enlace:")
    print("file://" + os.path.abspath("mapa_asignaciones_loc.html"))

    print("\nMapa generado y guardado como 'mapa_asignaciones_loc.html'.")

# ----- main -----

# prop = {
# array_of_file_json: ["paquetes_miraflores.json", "paquetes_miraflores_2.json"],
#}
def main(json_file_names, puntos_files_names):
    # 1. Obtener el grafo de Miraflores
    #print("Obteniendo grafo de Miraflores...")
    #graph = obtener_grafo_miraflores()
    #print("Grafo listo.")
    #print("")

    #if ver_grafo:
    #    visualizar_grafo(graph)
    #    mostrar_informacion_del_grafo(graph)

    # 2. Importar paquetes
    print("Importando paquetes...")
    paquetes = obtener_paquetes_con_coordenadas(json_file_names)
    for p in paquetes:
        if p['lat'] == None or p['lon'] == None:
            print("None para paquete con guia: ", p["numero_guia"])
            print(p)
        if p['lat'] == 0.0 or p['lon'] == 0.0:
            print("0.0 para paquete con guia: ", p["numero_guia"])
            print(p)
        if p['lat'] == 0 or p['lon'] == 0:
            print("0 para paquete con guia: ", p["numero_guia"])
            print(p)
    df_paquetes = pd.DataFrame([p for p in paquetes if p['lat'] != 0.0 and p['lon'] != 0.0])
    print("cantidad de paquetes: ", len(df_paquetes))
    #print("Paquetes importados:")
    #print(df_paquetes.head())
    #print("")

    camiones_disponibles = []

    for i in range(len(paquetes)//60):
        id_new_camion = f"T{i + 1}"
        new_camion = {'id': id_new_camion, 'max_weight': 900, 'max_volume': 5}
        camiones_disponibles.append(new_camion)

    # 2. Asignar paquetes a camiones
    print("Buscando asignación óptima de paquetes a camiones...")
    se_asigno = False

    while se_asigno == False:
        se_asigno = asignaciones_de_paquetes_a_grupos(df_paquetes, camiones_disponibles)

        if se_asigno == True:
            asignaciones_de_paquetes_a_locaciones(df_paquetes, puntos_files_names)

        if not se_asigno:
            id_new_camion = f"T{len(camiones_disponibles) + 1}"
            new_camion = {'id': id_new_camion, 'max_weight': 900, 'max_volume': 5}
            camiones_disponibles.append(new_camion)
            print("Intentando de nuevo...")

    print("Asignación de paquetes a camiones completada.")


if __name__ == "__main__":
    start_time = time.time()  # Inicio

    main(json_file_names=[
        "paquetes_miraflores.json",
        "paquetes_surquillo.json",
        "paquetes_santiago_de_surco.json",
        "paquetes_santa_anita.json",
        "paquetes_san_miguel.json",
        "paquetes_san_martin_de_porres.json",
        "paquetes_san_luis.json",
        "paquetes_san_isidro.json",
        "paquetes_san_borja.json",
        "paquetes_rimac.json",
        "paquetes_pueblo_libre.json",
        "paquetes_magdalena_del_mar.json",
        "paquetes_los_olivos.json",
        "paquetes_lince.json",
        "paquetes_la_victoria.json",
        "paquetes_la_molina.json",
        "paquetes_jesus_maria.json",
        "paquetes_independencia.json",
        "paquetes_el_agustino.json",
        "paquetes_chorrillos.json",
        "paquetes_barranco.json",
        "paquetes_breña.json",
    ], puntos_files_names=[
        "puntos_miraflores.json",
        "puntos_surquillo.json",
        "puntos_santiago_de_surco.json",
        "puntos_santa_anita.json",
        "puntos_san_miguel.json",
        "puntos_san_martin_de_porres.json",
        "puntos_san_luis.json",
        "puntos_san_isidro.json",
        "puntos_san_borja.json",
        "puntos_rimac.json",
        "puntos_pueblo_libre.json",
        "puntos_magdalena_del_mar.json",
        "puntos_los_olivos.json",
        "puntos_lince.json",
        "puntos_la_victoria.json",
        "puntos_la_molina.json",
        "puntos_jesus_maria.json",
        "puntos_independencia.json",
        "puntos_el_agustino.json",
        "puntos_chorrillos.json",
        "puntos_barranco.json",
        "puntos_breña.json",
    ])

    end_time = time.time()  # Fin

    elapsed_time = end_time - start_time
    print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos ({elapsed_time / 60:.2f} minutos)")



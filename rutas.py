import osmnx as ox
import os

import geopandas as gpd
import matplotlib.pyplot as plt

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from itertools import combinations

from scipy.spatial.distance import pdist, squareform

import folium

from typing import List, Dict
from ortools.linear_solver import pywraplp



# ----- Funciones -----

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

    print("Grafo listo para usar.")
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

def obtener_paquetes_con_coordenadas():
    with open("paquetes.json", "r") as f:
        paquetes = json.load(f)

    df = pd.read_excel("389657_21311_1740606018 (1).xlsx", engine="openpyxl")
    filtered_df = df[
        (df["LOCALIDAD"] == "MIRAFLORES - LIMA - LIMA") &
        (df["FECHA AO"] == "21/02/2025") &
        (df["FECHA VISITA 1"] == "22/02/2025") &
        (df["DESTINO"] == "LIM")
    ]
    # ahora solo obtener en este df el "PESO BALANZA", "PESO VOLUMEN" y "GUIA"
    filtered_df = filtered_df[["PESO BALANZA", "PESO VOLUMEN", "GUIA"]]
    #print primeros 10
    print("Primeros 10 paquetes:")
    print(filtered_df.head(10))



    paquetes_coordinadas_only = []

    for idx, paquete in enumerate(paquetes):
        coordenadas = paquete["coordenadas"]
        lat, lon = map(float, map(str.strip, coordenadas.split(",")))
        id = idx

        # find the paquete["numero_guia"] === filtered_df["GUIA"] and get the "PESO BALANZA" and "PESO VOLUMEN"
        peso_values = filtered_df[filtered_df["GUIA"] == paquete["numero_guia"]]["PESO BALANZA"].values
        # Verificar si hay valores antes de acceder al índice 0
        if len(peso_values) > 0:
            weight = peso_values[0]
        else:
            weight = None  # O algún valor por defecto

        peso_volumen_values = filtered_df[filtered_df["GUIA"] == paquete["numero_guia"]]["PESO VOLUMEN"].values
        if len(peso_volumen_values) > 0:
            volume = (peso_volumen_values[0] * 5000)/1000000
        else:
            volume = None

        # verify exist if not exist, then don't append
        if weight == None or volume == None:
            continue
        paquetes_coordinadas_only.append({ "id": id, "lat": lat, "lon": lon, "direccion": paquete["dir_calle"],  "weight": weight, "volume": volume })

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
                    'remaining_weight': truck['max_weight'] - cluster['total_weight'],
                    'remaining_volume': truck['max_volume'] - cluster['total_volume']
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


def capacitated_clara(packages: pd.DataFrame, trucks: List[Dict], n_samples=5, n_clusters=3):
    """
    Versión adaptada de CLARA para clustering con restricciones de capacidad.
    """
    best_clusters = None
    best_cost = float('inf')
    
    for _ in range(n_samples):
        # Muestra aleatoria del 80% de los datos
        sample = packages.sample(frac=0.8)
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

def bin_packing_split(cluster: pd.DataFrame, truck: Dict):
    """
    Divide un cluster usando algoritmo de bin packing 2D (peso + volumen).
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    items = [{'weight': row.weight, 'volume': row.volume} for _, row in cluster.iterrows()]
    
    # Variables de decisión
    x = {}
    for i in range(len(items)):
        for b in range(len(items)):
            x[i, b] = solver.IntVar(0, 1, f'x_{i}_{b}')
    
    # Restricciones
    for i in range(len(items)):
        solver.Add(sum(x[i, b] for b in range(len(items))) == 1)
    
    for b in range(len(items)):
        solver.Add(sum(items[i]['weight'] * x[i, b] for i in range(len(items))) <= truck['max_weight'])
        solver.Add(sum(items[i]['volume'] * x[i, b] for i in range(len(items))) <= truck['max_volume'])
    
    # Resolver
    status = solver.Solve()
    
    # Recuperar clusters
    clusters = []
    for b in range(len(items)):
        cluster_b = cluster.iloc[[i for i in range(len(items)) if x[i, b].solution_value() == 1]]
        if not cluster_b.empty:
            clusters.append({
                'total_weight': cluster_b['weight'].sum(),
                'total_volume': cluster_b['volume'].sum(),
                'packages': cluster_b
            })
    
    return clusters


def optimize_residual_space(assignments: List[Dict], trucks: List[Dict]):
    """
    Combina clusters pequeños para llenar espacios residuales.
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
                        
                        # Fusionar clusters
                        assignment['cluster']['packages'] = pd.concat([
                            assignment['cluster']['packages'],
                            candidate['cluster']['packages']
                        ])
                        assignment['remaining_weight'] -= candidate['cluster']['total_weight']
                        assignment['remaining_volume'] -= candidate['cluster']['total_volume']
                        assignments.remove(candidate)
                        break
    return assignments

# --- Modificar la función principal ---
def multi_capacity_clustering(packages, trucks, max_retries=5):
    trucks_sorted = sorted(trucks, key=lambda x: (-x['max_weight'], -x['max_volume']))
    best_assignment = None
    min_trucks = float('inf')
    
    for _ in range(max_retries):
        # Usar CLARA en lugar de K-means puro
        clusters = capacitated_clara(packages, trucks, n_samples=5, n_clusters=len(trucks))
        
        # Asignar a camiones
        assignments, unassigned = assign_to_trucks(clusters, trucks_sorted)
        
        # Optimizar espacios residuales
        assignments = optimize_residual_space(assignments, trucks_sorted)
        
        if len(assignments) < min_trucks and not unassigned:
            min_trucks = len(assignments)
            best_assignment = assignments
    
    return best_assignment



def plot_assignments_on_map(assignments):
    # Crear un mapa centrado en Miraflores, Lima
    map_center = [-12.111, -77.031]  # Coordenadas de Miraflores
    m = folium.Map(location=map_center, zoom_start=14)

    # Colores para diferenciar camiones
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 
              'lightgreen', 'gray', 'black', 'lightgray']

    # Añadir marcadores para cada camión y sus paquetes
    for i, assign in enumerate(assignments):
        truck_id = assign['truck_id']
        cluster = assign['cluster']
        color = colors[i % len(colors)]  # Asignar un color único por camión

        # Añadir marcador para el camión (punto de partida)
        folium.Marker(
            location=[cluster['packages']['lat'].mean(), cluster['packages']['lon'].mean()],
            popup=f"Camion {truck_id}",
            icon=folium.Icon(color=color, icon='truck', prefix='fa')
        ).add_to(m)

        # Añadir marcadores para los paquetes del camión
        for _, row in cluster['packages'].iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"Paquete: {row['weight']} kg, {row['volume']} ton",
                icon=folium.Icon(color=color, icon='box', prefix='fa')
            ).add_to(m)

    return m


def asignaciones_de_paquetes_a_grupos(packages, trucks):
    # Generar asignación óptima
    assignments = multi_capacity_clustering(packages, trucks)
    
    # Mostrar resultados
    if assignments:
        print("Asignación de camiones:")
        for assign in assignments:
            print(f"\nCamión {assign['truck_id']}:")
            print(assign['cluster']['packages'][['weight', 'volume', 'lat', 'lon']])
            print(f"Total peso: {assign['cluster']['total_weight']} kg")
            print(f"Total volumen: {assign['cluster']['total_volume']} m³")
            print(f"Capacidad restante: {assign['remaining_weight']} kg, {assign['remaining_volume']} m³")

        print()
        total_weight_used = sum(a['cluster']['total_weight'] for a in assignments)
        total_weight_desperdiciado = sum(a['remaining_weight'] for a in assignments)
        total_volume_used = sum(a['cluster']['total_volume'] for a in assignments)
        total_volume_desperdiciado = sum(a['remaining_volume'] for a in assignments)

        print(f"Total peso usado: {total_weight_used} kg ({
            total_weight_used / (total_weight_used + total_weight_desperdiciado) * 100:.2f
            }%) vs sin usar: {total_weight_desperdiciado} kg ({total_weight_desperdiciado / (total_weight_used + total_weight_desperdiciado) * 100:.2f}%)")

        print(f"Total peso usado: {total_volume_used} kg ({
            total_volume_used  / (total_volume_used + total_volume_desperdiciado) * 100:.2f
            }%) vs sin usar: {total_volume_desperdiciado} kg ({total_volume_desperdiciado / (total_volume_used + total_volume_desperdiciado) * 100:.2f}%)")
        print(f"Total camiones usados: {len(assignments)}")

        # Visualizar en el mapa
        map_assignments = plot_assignments_on_map(assignments)
        map_assignments.save("mapa_asignaciones.html")  # Guardar el mapa en un archivo HTML
        print("\nMapa generado y guardado como 'mapa_asignaciones.html'.")
        return True
    else:
        print("No se encontró una asignación válida.")
        return False



# ----- main -----

def main(ver_grafo=False):
    # 1. Obtener el grafo de Miraflores
    graph = obtener_grafo_miraflores()

    if ver_grafo:
        visualizar_grafo(graph)
        mostrar_informacion_del_grafo(graph)

    # 2. Importar paquetes
    paquetes = obtener_paquetes_con_coordenadas()
    for p in paquetes:
        # if paquetes.lan o lon == 0.0, then print it
        if p['lat'] == None or p['lon'] == None:
            print(p)
    df_paquetes = pd.DataFrame([p for p in paquetes if p['lat'] != 0.0 and p['lon'] != 0.0])
    print("Dimensiones de packages:", df_paquetes.shape)
    print("Primeras filas de packages:\n", df_paquetes.head())

    camiones_disponibles = [
        {'id': 'T1', 'max_weight': 900, 'max_volume': 12},
        {'id': 'T2', 'max_weight': 900, 'max_volume': 12},
        {'id': 'T3', 'max_weight': 900, 'max_volume': 12},
        {'id': 'T4', 'max_weight': 900, 'max_volume': 12},
    ]

    # 2. Asignar paquetes a camiones
    se_asigno = False
    while se_asigno == False:
        id_new_camion = f"T{len(camiones_disponibles) + 1}"
        new_camion = {'id': id_new_camion, 'max_weight': 900, 'max_volume': 12}
        camiones_disponibles.append(new_camion)

        se_asigno = asignaciones_de_paquetes_a_grupos(df_paquetes, camiones_disponibles)

        if not se_asigno:
            print("Intentando de nuevo...")



if __name__ == "__main__":
    main(ver_grafo=False)

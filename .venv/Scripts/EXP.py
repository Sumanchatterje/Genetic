import random
import numpy as np
import networkx as nx
import osmnx as ox
import folium
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import os

print("Starting program...")

LOCATIONS = {
    "Start": (22.609372, 88.422594),
    "Midpoint": (22.611263, 88.425738),
    "End": (22.623395, 88.417155)
}


def get_osm_graph():
    graph_file = "road_network.graphml"
    if not os.path.exists(graph_file):
        print("Downloading OSM graph...")
        try:
            graph = ox.graph_from_point(LOCATIONS["Start"], dist=10000, network_type='drive')
            ox.save_graphml(graph, graph_file)
        except Exception as e:
            print(f"Error downloading graph: {e}")
            raise
    else:
        print("Loading OSM graph from file...")
        try:
            graph = ox.load_graphml(graph_file)
        except Exception as e:
            print(f"Error loading graph: {e}")
            raise
    print("Graph loaded successfully")
    return graph


def get_nearest_nodes(graph, locations):
    """Get nearest nodes in the graph for each location"""
    return {name: ox.nearest_nodes(graph, loc[1], loc[0])
            for name, loc in locations.items()}


def get_route_coords(graph, start_node, end_node):
    """Get the coordinates for each node in the route"""
    try:
        # Get the shortest path between nodes
        route = nx.shortest_path(graph, start_node, end_node, weight='length')

        # Extract the coordinates for each node in the path
        path_coords = []
        for node in route:
            # Get node attributes
            node_data = graph.nodes[node]
            path_coords.append((node_data['y'], node_data['x']))

        return path_coords
    except nx.NetworkXNoPath:
        print(f"No path found between nodes {start_node} and {end_node}")
        return None


def plot_route_folium(graph, best_route):
    # Create a map centered at the start location
    m = folium.Map(location=LOCATIONS["Start"], zoom_start=14)

    # Colors for different segments
    colors = ['blue', 'red', 'green']

    # Add markers for all locations
    for name, coords in LOCATIONS.items():
        folium.Marker(
            coords,
            popup=name,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

    # Get nearest nodes for all locations
    nearest_nodes = get_nearest_nodes(graph, LOCATIONS)

    # Plot routes between consecutive points
    location_names = list(LOCATIONS.keys())
    for i in range(len(best_route) - 1):
        # Get start and end points for this segment
        start_name = location_names[best_route[i]]
        end_name = location_names[best_route[i + 1]]

        # Get route coordinates
        route_coords = get_route_coords(
            graph,
            nearest_nodes[start_name],
            nearest_nodes[end_name]
        )

        if route_coords:
            # Draw the route segment
            folium.PolyLine(
                route_coords,
                weight=4,
                color=colors[i % len(colors)],
                opacity=0.8,
                popup=f'Segment {i + 1}: {start_name} to {end_name}'
            ).add_to(m)

    m.save("delivery_route.html")
    print("Interactive map saved as 'delivery_route.html'")


def compute_real_distance_matrix(locations, graph):
    print("Computing distance matrix...")
    try:
        nodes = get_nearest_nodes(graph, locations)
        size = len(nodes)
        matrix = np.zeros((size, size))
        names = list(nodes.keys())

        for i in range(size):
            for j in range(size):
                if i != j:
                    try:
                        route_length = nx.shortest_path_length(
                            graph,
                            nodes[names[i]],
                            nodes[names[j]],
                            weight='length'
                        )
                        matrix[i][j] = route_length / 1000  # Convert to km
                    except nx.NetworkXNoPath:
                        print(f"No path found between {names[i]} and {names[j]}")
                        matrix[i][j] = float('inf')

        print("Distance matrix computed successfully")
        return matrix
    except Exception as e:
        print(f"Error computing distance matrix: {e}")
        raise


# Rest of the genetic algorithm setup remains the same
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def create_individual():
    return [1]


toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    route = [0] + individual + [2]
    total_distance = sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    return (total_distance,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    print("Starting main optimization...")
    graph = get_osm_graph()
    global distance_matrix
    distance_matrix = compute_real_distance_matrix(LOCATIONS, graph)

    population = toolbox.population(n=10)

    for ind in population:
        ind.fitness.values = evaluate(ind)

    best_ind = tools.selBest(population, 1)[0]
    best_route = [0] + list(best_ind) + [2]

    print("\nResults:")
    print("Optimal Route:", [list(LOCATIONS.keys())[i] for i in best_route])
    print("Total Distance: {:.2f} km".format(evaluate(best_ind)[0]))

    print("\nCreating map with road network path...")
    plot_route_folium(graph, best_route)


if __name__ == "__main__":
    print("\nProgram started from __main__")
    main()
    print("\nProgram completed successfully")
import random
import numpy as np
import networkx as nx
import osmnx as ox
import folium
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import os

print("Starting program...")

# Hardcoded locations (Latitude, Longitude)
LOCATIONS = {
    "Start": (22.740697, 88.457492),
    "Midpoint": (22.724759, 88.479110),
    "End": (22.719399, 88.486519)
}

print("Locations loaded:", LOCATIONS)


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


try:
    print("Getting graph...")
    graph = get_osm_graph()
except Exception as e:
    print(f"Failed to get graph: {e}")
    raise


def compute_real_distance_matrix(locations, graph):
    print("Computing distance matrix...")
    try:
        nodes = {name: ox.nearest_nodes(graph, loc[1], loc[0]) for name, loc in locations.items()}
        size = len(nodes)
        matrix = np.zeros((size, size))
        names = list(nodes.keys())

        for i in range(size):
            for j in range(size):
                if i != j:
                    route_length = nx.shortest_path_length(graph, nodes[names[i]], nodes[names[j]], weight='length')
                    matrix[i][j] = route_length / 1000  # Convert to km
        print("Distance matrix computed successfully")
        return matrix
    except Exception as e:
        print(f"Error computing distance matrix: {e}")
        raise


try:
    distance_matrix = compute_real_distance_matrix(LOCATIONS, graph)
    print("Distance matrix shape:", distance_matrix.shape)
except Exception as e:
    print(f"Failed to compute distance matrix: {e}")
    raise

# Clear any existing DEAP creators to avoid duplicates
if 'FitnessMin' in creator.__dict__:
    del creator.FitnessMin
if 'Individual' in creator.__dict__:
    del creator.Individual

print("Setting up genetic algorithm...")
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# Modified for small routes
def create_individual():
    # For small routes, we can just return a fixed sequence
    # Since we only have 3 points (Start, Midpoint, End),
    # and Start/End are fixed, we only need to handle the Midpoint
    return [1]  # Midpoint is always at index 1


toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    route = [0] + individual + [2]  # Start (0) -> Midpoint (1) -> End (2)
    total_distance = sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    return (total_distance,)


def custom_mutation(individual, indpb):
    """Custom mutation for small routes - does nothing since we only have one possible route"""
    return individual,


def custom_crossover(ind1, ind2):
    """Custom crossover for small routes - simply returns the same individuals"""
    return ind1, ind2


toolbox.register("evaluate", evaluate)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", custom_mutation, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    print("Starting main optimization...")
    population = toolbox.population(n=10)  # Reduced population size since we have only one possible route

    ngen, cxpb, mutpb = 10, 0.7, 0.2  # Reduced generations since we have only one possible route

    print(f"Running genetic algorithm with:")
    print(f"Population size: 10")
    print(f"Generations: {ngen}")
    print(f"Crossover probability: {cxpb}")
    print(f"Mutation probability: {mutpb}")

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min"]

    # Single evaluation for all individuals since route is fixed
    for ind in population:
        ind.fitness.values = evaluate(ind)

    best_ind = tools.selBest(population, 1)[0]
    best_route = [0] + list(best_ind) + [2]

    print("\nResults:")
    print("Optimal Route:", [list(LOCATIONS.keys())[i] for i in best_route])
    print("Total Distance: {:.2f} km".format(evaluate(best_ind)[0]))

    # Plot route
    print("\nCreating map...")
    plot_route_folium(best_route)


def plot_route_folium(best_route):
    m = folium.Map(location=LOCATIONS["Start"], zoom_start=14)

    # Add markers for the locations
    for name, coords in LOCATIONS.items():
        folium.Marker(coords, popup=name).add_to(m)

    points = [LOCATIONS[list(LOCATIONS.keys())[i]] for i in best_route]
    folium.PolyLine(points, color='blue', weight=5, opacity=0.7).add_to(m)

    m.save("displacement_delivery_route.html")
    print("Interactive map saved as 'delivery_route.html'")


if __name__ == "__main__":
    print("\nProgram started from __main__")
    main()
    print("\nProgram completed successfully")
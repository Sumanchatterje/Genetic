import random
import numpy as np
import networkx as nx
import osmnx as ox
import folium
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import os
from datetime import datetime
from folium.plugins import PolyLineTextPath

print("Starting program...")

# Define delivery locations - can be expanded with more waypoints
LOCATIONS = {
    "Start": (22.609372, 88.422594),
    "Midpoint": (22.611263, 88.425738),
    "End": (22.623395, 88.417155),
    # You can add more waypoints here as needed
}

def get_osm_graph(center_point, dist=5000, network_type='drive', force_download=False, simplify=True):
    """
    Get or download an OSM graph for routing

    Parameters:
    - center_point: (lat, lon) tuple to center the graph
    - dist: distance in meters to download around the center point
    - network_type: type of network ('drive', 'walk', 'bike', etc.)
    - force_download: if True, download even if file exists
    - simplify: whether to simplify the graph topology
    """
    graph_file = f"road_network_{network_type}_{dist}m.graphml"

    if force_download or not os.path.exists(graph_file):
        print(f"Downloading OSM graph ({network_type} network, {dist}m radius)...")
        try:
            # Download the graph - we'll keep it in WGS84 coordinates for compatibility
            # Setting simplify=False to keep all nodes and get more accurate routes
            graph = ox.graph_from_point(center_point, dist=dist, network_type=network_type, simplify=simplify)
            # Save graph for future use
            ox.save_graphml(graph, graph_file)
            print(f"Graph downloaded and saved to {graph_file}")
        except Exception as e:
            print(f"Error downloading graph: {e}")
            raise
    else:
        print(f"Loading OSM graph from {graph_file}...")
        try:
            graph = ox.load_graphml(graph_file)
        except Exception as e:
            print(f"Error loading graph: {e}")
            raise

    print(f"Graph loaded successfully with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

def get_nearest_nodes(graph, locations):
    """Get nearest nodes in the graph for each location"""
    nearest_nodes = {}
    for name, loc in locations.items():
        try:
            # Get the nearest node to this location
            nearest = ox.distance.nearest_nodes(graph, loc[1], loc[0])
            nearest_nodes[name] = nearest
            print(f"Found nearest node for {name}: {nearest}")
        except Exception as e:
            print(f"Error finding nearest node for {name}: {e}")
            raise
    return nearest_nodes

def get_route(graph, start_node, end_node, weight='length'):
    """Get the route between two nodes using Dijkstra's algorithm"""
    try:
        # Get the shortest path between nodes
        route = nx.shortest_path(graph, start_node, end_node, weight=weight)
        return route
    except nx.NetworkXNoPath:
        print(f"No path found between nodes {start_node} and {end_node}")
        return None
    except Exception as e:
        print(f"Error finding route: {e}")
        return None

def get_route_coords(graph, route):
    """Extract coordinates from a route"""
    if not route:
        return None

    # Extract the coordinates for each node in the path
    path_coords = []
    for node in route:
        # Get node attributes
        node_data = graph.nodes[node]
        path_coords.append((node_data['y'], node_data['x']))

    return path_coords

def calculate_route_length(graph, route):
    """Calculate the total length of a route"""
    if not route or len(route) < 2:
        return 0

    total_length = 0
    for i in range(len(route) - 1):
        # Get the edge data between these nodes
        try:
            u, v = route[i], route[i + 1]
            # Check if there's an edge between these nodes
            if graph.has_edge(u, v):
                edge_data = graph.get_edge_data(u, v)
                # There might be multiple edges between two nodes, take the shortest
                if edge_data:
                    # Get the edge with minimum length
                    min_length = min(data.get('length', float('inf')) for data in edge_data.values())
                    total_length += min_length
            else:
                print(f"Warning: No edge between nodes {u} and {v}")
        except Exception as e:
            print(f"Error calculating edge length: {e}")

    return total_length / 1000  # Convert to km

def plot_route_folium(graph, best_route, locations, filename="delivery_route.html"):
    """Create an interactive folium map showing the optimized route"""
    # Get a list of location names
    location_names = list(locations.keys())

    # Calculate the center of all locations for the map center
    lats = [loc[0] for loc in locations.values()]
    lons = [loc[1] for loc in locations.values()]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Create a map centered at the average location - using only OpenStreetMap
    m = folium.Map(location=(center_lat, center_lon), zoom_start=15,
                   tiles="OpenStreetMap")

    # Add a title to the map
    title_html = '''
        <h3 align="center" style="font-size:18px; background-color: rgba(255,255,255,0.8); 
        padding: 10px; border-radius: 5px; margin: 10px;"><b>Optimized Delivery Route</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Use a single consistent color for the main route
    route_color = '#FF4500'  # Orange-Red

    # Get nearest nodes for all locations
    nearest_nodes = get_nearest_nodes(graph, locations)

    # Add markers for all locations with custom icons
    icons = {
        'Start': 'play',
        'End': 'flag-checkered',
        'Midpoint': 'map-pin'
    }

    # Use custom markers with more visibility
    for name, coords in locations.items():
        # Choose icon based on the location name
        icon_name = icons.get(name, 'info-sign')

        # Choose color based on position
        if name == "Start":
            color = 'green'
        elif name == "End":
            color = 'red'
        else:
            color = 'blue'

        # Create a circular marker with popup
        folium.CircleMarker(
            location=coords,
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(f"<b>{name}</b>", max_width=200),
            tooltip=name
        ).add_to(m)

        # Add a more distinctive marker as well
        folium.Marker(
            coords,
            popup=f"<b>{name}</b><br>Coordinates: {coords[0]:.6f}, {coords[1]:.6f}",
            tooltip=name,
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa')
        ).add_to(m)

    # Total distance for summary
    total_distance = 0
    segments_info = []

    # Collect all route coordinates to create a single smooth line
    all_route_coords = []

    # Calculate and collect coordinates for each segment
    for i in range(len(best_route) - 1):
        # Get start and end points for this segment
        start_idx = best_route[i]
        end_idx = best_route[i + 1]

        start_name = location_names[start_idx]
        end_name = location_names[end_idx]

        # Get the nodes for this segment
        start_node = nearest_nodes[start_name]
        end_node = nearest_nodes[end_name]

        # Check if nodes are valid
        if start_node is None or end_node is None:
            continue

        # Get the route between these nodes
        route_nodes = get_route(graph, start_node, end_node)
        if route_nodes:
            # Get coordinates for this route
            route_coords = get_route_coords(graph, route_nodes)

            if route_coords:
                # Calculate segment distance
                segment_distance = calculate_route_length(graph, route_nodes)
                total_distance += segment_distance
                segments_info.append(f"{start_name} → {end_name}: {segment_distance:.2f} km")

                # Add to our complete route coordinates
                all_route_coords.extend(route_coords)
        else:
            # Use straight line as fallback
            lat1, lon1 = locations[start_name]
            lat2, lon2 = locations[end_name]

            from math import radians, cos, sin, asin, sqrt
            def haversine(lon1, lat1, lon2, lat2):
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                km = 6367 * c
                return km

            segment_distance = haversine(lon1, lat1, lon2, lat2)
            total_distance += segment_distance
            segments_info.append(f"{start_name} → {end_name}: {segment_distance:.2f} km")

            # Direct line between points
            all_route_coords.append((lat1, lon1))
            all_route_coords.append((lat2, lon2))

    # Plot the optimized route with a polyline
    folium.PolyLine(
        all_route_coords, color=route_color, weight=5, opacity=0.8
    ).add_to(m)

    # Add segment info to popup text
    route_summary = f"Total Distance: {total_distance:.2f} km"
    for segment in segments_info:
        route_summary += f"\n{segment}"

    folium.Marker(
        location=all_route_coords[len(all_route_coords)//2],
        popup=route_summary,
        icon=folium.Icon(color='blue')
    ).add_to(m)

    # Save the route to an HTML file
    m.save(filename)
    print(f"Map saved as {filename}")

# Set up genetic algorithm for route optimization
def setup_genetic_algorithm(num_waypoints):
    # Set up Genetic Algorithm components
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Define how to create a single individual
    def create_permutation():
        if num_waypoints <= 3:
            return [1]  # Middle point is always at index 1
        # Otherwise create a permutation of all waypoints except the start and end points
        return random.sample(range(1, num_waypoints - 1), num_waypoints - 2)

    # Initialize toolbox components
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, create_permutation)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function (fitness) and mutation/crossover operations
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_individual)

    return toolbox


def evaluate_individual(individual):
    """Evaluate the fitness of an individual (i.e., the total route distance)"""
    total_distance = 0
    for i in range(len(individual) - 1):
        # Get the distance between two consecutive locations
        start = individual[i]
        end = individual[i + 1]
        total_distance += distance_matrix[start][end]

    return total_distance,


def run_genetic_algorithm(num_generations=100, pop_size=50, cx_prob=0.7, mut_prob=0.2):
    """Run the genetic algorithm to optimize the delivery route"""
    print("Setting up genetic algorithm...")

    # Generate initial population
    toolbox = setup_genetic_algorithm(len(LOCATIONS))
    population = toolbox.population(n=pop_size)

    # Create a hall of fame to track the best solutions
    hof = tools.HallOfFame(1)

    # Set up statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the algorithm
    print("Running genetic algorithm...")
    algorithms.eaSimple(population, toolbox, cxpb=cx_prob, mutpb=mut_prob,
                        ngen=num_generations, stats=stats, halloffame=hof,
                        verbose=True)

    print(f"Best route found: {hof[0]}")
    return hof[0]


# Get the OSM graph for the region around the first location (Start)
graph = get_osm_graph(center_point=LOCATIONS["Start"])

# Compute the distance matrix for the locations
distance_matrix = compute_distance_matrix(LOCATIONS, graph)

# Run the genetic algorithm to find the best route
best_route = run_genetic_algorithm()

# Plot the best route on a map
plot_route_folium(graph, best_route, LOCATIONS)

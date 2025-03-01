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

    # Create a map centered at the average location - using a more detailed map tile
    m = folium.Map(location=(center_lat, center_lon), zoom_start=15,
                   tiles="cartodbpositron")  # More detailed Carto tiles

    # Add layer control and multiple map options
    folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.TileLayer(
        'Stamen Terrain',
        name='Terrain',
        attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.'
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # Add a title to the map
    title_html = '''
        <h3 align="center" style="font-size:18px; background-color: rgba(255,255,255,0.8); 
        padding: 10px; border-radius: 5px; margin: 10px;"><b>Optimized Delivery Route</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # More distinctive colors with higher contrast
    colors = ['#0000FF', '#FF0000', '#00CC00', '#AA00AA', '#FF8800', '#00CCCC']

    # Larger, more distinctive icons
    icon_size = (35, 35)  # Larger icons

    # Get nearest nodes for all locations
    nearest_nodes = get_nearest_nodes(graph, locations)

    # Debug: Print nearest nodes
    print("\nNearest nodes for locations:")
    for name, node_id in nearest_nodes.items():
        print(f"{name}: {node_id}")

    # Add markers for all locations with custom icons
    icons = {
        'Start': 'play',
        'End': 'flag-checkered',
        'Midpoint': 'map-pin'
    }

    # Use custom markers with more visibility
    for i, (name, coords) in enumerate(locations.items()):
        # Choose icon based on the location name
        icon_name = icons.get(name, 'info-sign')

        # Choose color based on position
        if name == "Start":
            color = 'green'
            icon_color = 'white'
        elif name == "End":
            color = 'red'
            icon_color = 'white'
        else:
            color = 'blue'
            icon_color = 'white'

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

    # Calculate and display the route for each segment of the best route
    for i in range(len(best_route) - 1):
        # Get start and end points for this segment
        start_idx = best_route[i]
        end_idx = best_route[i + 1]

        start_name = location_names[start_idx]
        end_name = location_names[end_idx]

        print(f"\nProcessing route segment: {start_name} to {end_name}")

        # Get the nodes for this segment
        start_node = nearest_nodes[start_name]
        end_node = nearest_nodes[end_name]

        # Check if nodes are valid
        if start_node is None or end_node is None:
            print(f"Warning: Invalid nodes for segment {start_name} to {end_name}")
            continue

        # Get the route between these nodes
        route_nodes = get_route(graph, start_node, end_node)
        if route_nodes:
            print(f"Found route with {len(route_nodes)} nodes")

            # Get coordinates for this route
            route_coords = get_route_coords(graph, route_nodes)

            if route_coords:
                # Calculate segment distance
                segment_distance = calculate_route_length(graph, route_nodes)
                print(f"Segment distance: {segment_distance:.2f} km")

                total_distance += segment_distance
                segments_info.append(f"{start_name} → {end_name}: {segment_distance:.2f} km")

                # Draw the route segment with a thicker line and arrow decoration
                path = folium.PolyLine(
                    route_coords,
                    weight=6,  # Thicker line
                    color=colors[i % len(colors)],
                    opacity=0.9,  # More opaque
                    tooltip=f"Segment {i + 1}: {start_name} to {end_name} ({segment_distance:.2f} km)",
                    smoothFactor=1
                ).add_to(m)

                # Add arrows to show direction
                folium.plugins.PolyLineTextPath(
                    path,
                    text="→",
                    repeat=True,
                    offset=8,
                    attributes={"fill": "#FFFFFF", "font-weight": "bold", "font-size": "24"}
                ).add_to(m)

                # Add distance label
                midpoint_idx = len(route_coords) // 2
                if midpoint_idx < len(route_coords):
                    midpoint = route_coords[midpoint_idx]
                    folium.map.Marker(
                        midpoint,
                        icon=folium.DivIcon(
                            icon_size=(150, 20),
                            icon_anchor=(75, 10),
                            html=f'<div style="background-color: white; width: 100px; text-align: center; '
                                 f'border-radius: 10px; padding: 2px 5px; font-weight: bold; opacity: 0.8;">'
                                 f'{segment_distance:.2f} km</div>'
                        )
                    ).add_to(m)
            else:
                print(f"Warning: Could not extract coordinates for route")
        else:
            print(f"Warning: No route found between {start_name} and {end_name}")

            # Try to add a straight line as fallback
            fallback_coords = [
                (locations[start_name][0], locations[start_name][1]),
                (locations[end_name][0], locations[end_name][1])
            ]

            # Calculate straight-line distance (haversine formula)
            from math import radians, cos, sin, asin, sqrt

            def haversine(lon1, lat1, lon2, lat2):
                """Calculate the great circle distance between two points"""
                # Convert decimal degrees to radians
                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

                # Haversine formula
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                r = 6371  # Radius of earth in kilometers
                return c * r

            # Calculate the straight-line distance
            lat1, lon1 = locations[start_name]
            lat2, lon2 = locations[end_name]
            segment_distance = haversine(lon1, lat1, lon2, lat2)

            total_distance += segment_distance
            segments_info.append(f"{start_name} → {end_name}: {segment_distance:.2f} km (straight line)")

            # Draw a dashed straight line
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                weight=4,
                color=colors[i % len(colors)],
                opacity=0.8,
                dash_array='5, 10',
                tooltip=f"Segment {i + 1}: {start_name} to {end_name} ({segment_distance:.2f} km - approximate)"
            ).add_to(m)

    # Add a legend with route information
    legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; width: 250px; 
        height: auto; background-color: white; border:2px solid grey; z-index:9999; 
        font-size:14px; padding: 15px; border-radius: 10px; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
        <h4 style="margin-top: 0; color: #333;"><b>Route Summary</b></h4>
        <p><b>Total Distance:</b> {total_distance:.2f} km</p>
        <hr style="margin: 5px 0;">
        <p><b>Segments:</b></p>
        <ol style="padding-left: 20px; margin-top: 5px;">{"".join([f"<li style='margin-bottom: 8px;'>{info}</li>" for info in segments_info])}</ol>
        <hr style="margin: 5px 0;">
        <p style="font-size: 12px; color: #777; margin-bottom: 0;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    m.save(filename)
    print(f"Interactive map saved as '{filename}'")

    return total_distance


def compute_distance_matrix(locations, graph):
    """Compute a matrix of real-world distances between all locations"""
    print("\nComputing distance matrix...")
    try:
        # Get the nearest nodes for each location
        nodes = get_nearest_nodes(graph, locations)
        size = len(nodes)
        matrix = np.zeros((size, size))
        names = list(nodes.keys())

        # For each pair of locations
        for i in range(size):
            for j in range(size):
                if i != j:
                    try:
                        # Get route between locations
                        route_nodes = get_route(
                            graph,
                            nodes[names[i]],
                            nodes[names[j]]
                        )

                        if route_nodes:
                            # Calculate the route length
                            distance = calculate_route_length(graph, route_nodes)
                            matrix[i][j] = distance
                        else:
                            # If no route found, use straight-line distance as fallback
                            from math import radians, cos, sin, asin, sqrt

                            def haversine(lon1, lat1, lon2, lat2):
                                """Calculate the great circle distance between two points"""
                                # Convert decimal degrees to radians
                                lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

                                # Haversine formula
                                dlon = lon2 - lon1
                                dlat = lat2 - lat1
                                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                                c = 2 * asin(sqrt(a))
                                r = 6371  # Radius of earth in kilometers
                                return c * r

                            # Calculate the straight-line distance
                            lat1, lon1 = locations[names[i]]
                            lat2, lon2 = locations[names[j]]
                            distance = haversine(lon1, lat1, lon2, lat2)

                            print(
                                f"No route found between {names[i]} and {names[j]}, using straight-line distance: {distance:.2f} km")
                            matrix[i][j] = distance
                    except Exception as e:
                        print(f"Error computing distance between {names[i]} and {names[j]}: {e}")

                        # Use a placeholder value that won't break the algorithm
                        matrix[i][j] = 999.99

        # Make sure we don't have any infinite or zero distances where we shouldn't
        matrix[matrix == float('inf')] = 999.99
        for i in range(size):
            for j in range(size):
                if i != j and matrix[i][j] == 0:
                    # Use a small default value
                    matrix[i][j] = 0.01

        print("Distance matrix computed successfully")

        # Print the distance matrix for debugging
        print("\nDistance Matrix (km):")
        print("     " + "  ".join(f"{name:<8}" for name in names))
        for i, name in enumerate(names):
            print(f"{name:<5}" + "  ".join(f"{matrix[i][j]:<8.2f}" for j in range(size)))

        return matrix
    except Exception as e:
        print(f"Error computing distance matrix: {e}")
        raise


def setup_genetic_algorithm(num_waypoints, pop_size=50):
    """Set up the genetic algorithm components"""
    # Clean up any existing definitions to avoid warnings
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin
    if 'Individual' in creator.__dict__:
        del creator.Individual

    # Create fitness and individual classes
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize toolbox
    toolbox = base.Toolbox()

    # Define how to create a single individual
    def create_permutation():
        # For just 3 locations, there's only one possible order
        if num_waypoints <= 3:
            return [1]  # Middle point is always at index 1
        # Otherwise create a permutation of all waypoints (excluding start and end)
        return random.sample(range(1, num_waypoints - 1), num_waypoints - 2)

    # Register the creation methods
    toolbox.register("indices", create_permutation)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def evaluate(individual, distance_matrix, fixed_start=0, fixed_end=None):
    """
    Evaluate the fitness of an individual

    Parameters:
    - individual: a permutation of waypoints
    - distance_matrix: matrix of distances between locations
    - fixed_start: index of the fixed start location (default 0)
    - fixed_end: index of the fixed end location (default None, will use last location)
    """
    if fixed_end is None:
        fixed_end = len(distance_matrix) - 1

    # Complete route with fixed start and end
    route = [fixed_start] + list(individual) + [fixed_end]

    # Calculate total distance
    total_distance = 0
    for i in range(len(route) - 1):
        segment_distance = distance_matrix[route[i]][route[i + 1]]
        total_distance += segment_distance

    return (total_distance,)


def run_genetic_algorithm(toolbox, distance_matrix, pop_size=50, gen_count=40,
                          cx_prob=0.8, mut_prob=0.2, fixed_start=0, fixed_end=None):
    """Run the genetic algorithm to find the optimal route"""
    if fixed_end is None:
        fixed_end = len(distance_matrix) - 1

    # Set up the evaluation function
    toolbox.register("evaluate", evaluate, distance_matrix=distance_matrix,
                     fixed_start=fixed_start, fixed_end=fixed_end)

    # Set up genetic operators
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create initial population
    pop = toolbox.population(n=pop_size)

    # Keep track of the best individual
    hof = tools.HallOfFame(1)

    # Set up statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Run the genetic algorithm
    print(f"\nRunning genetic algorithm with population size {pop_size} for {gen_count} generations...")

    # Run the algorithm
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob,
                                       ngen=gen_count, stats=stats, halloffame=hof, verbose=True)

    # Extract the best solution
    best_ind = hof[0]
    best_route = [fixed_start] + list(best_ind) + [fixed_end]
    best_fitness = best_ind.fitness.values[0]

    # Extract statistics
    gen = logbook.select("gen")
    fit_mins = logbook.select("min")
    fit_avgs = logbook.select("avg")

    # Plot the evolution of fitness
    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_mins, 'b-', label='Minimum Fitness')
    plt.plot(gen, fit_avgs, 'r-', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Distance in km)')
    plt.title('Evolution of Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_evolution.png')
    print("Fitness evolution plot saved as 'fitness_evolution.png'")

    return best_route, best_fitness, gen, fit_mins, fit_avgs


def main():
    print("Starting main optimization...")

    # Define parameters
    center_point = LOCATIONS["Start"]  # Center the graph around the start point
    graph_radius = 10000  # 10km radius - increased for better coverage

    # Get the road network graph - force download to ensure we have the latest data
    # Setting simplify=False to preserve more road network detail
    graph = get_osm_graph(center_point, dist=graph_radius, network_type='drive',
                          force_download=True, simplify=False)

    # Compute the distance matrix
    distance_matrix = compute_distance_matrix(LOCATIONS, graph)

    # Set up the genetic algorithm
    num_locations = len(LOCATIONS)
    toolbox = setup_genetic_algorithm(num_locations)

    # If there are only three locations, we have a single solution
    if num_locations <= 3:
        print("\nOnly 3 locations detected, using direct route...")
        best_route = list(range(num_locations))
        best_fitness = evaluate([1], distance_matrix, fixed_start=0, fixed_end=2)[0]
        print(f"Simple route distance: {best_fitness:.2f} km")
    else:
        # Run the genetic algorithm
        best_route, best_fitness, gen, fit_mins, fit_avgs = run_genetic_algorithm(
            toolbox, distance_matrix, pop_size=50, gen_count=40)

    # Print the results
    print("\nOptimization Results:")
    print("Optimal Route:", [list(LOCATIONS.keys())[i] for i in best_route])
    print("Total Distance: {:.2f} km".format(best_fitness))

    # Create the interactive map
    print("\nCreating interactive map with optimized route...")
    total_distance = plot_route_folium(graph, best_route, LOCATIONS)

    print(f"\n! Total route distance: {total_distance:.2f} km")
    print("Interactive map saved as 'delivery_route.html'")


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\nProgram started at {start_time}")

    try:
        main()
        end_time = datetime.now()
        run_time = (end_time - start_time).total_seconds()
        print(f"\nProgram completed successfully in {run_time:.2f} seconds")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
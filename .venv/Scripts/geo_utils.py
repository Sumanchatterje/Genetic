# geo_utils.py
import openrouteservice
import time
from geopy.geocoders import Nominatim
import json
import os

# API key for OpenRouteService
API_KEY = "5b3ce3597851110001cf6248e8c26805c2cc4297aa6123e870e315a1"  # Replace with your OpenRouteService key
client = openrouteservice.Client(key=API_KEY)
geolocator = Nominatim(user_agent="delivery_route_optimizer", timeout=10)


def geocode_location(address):
    """
    Convert an address string to (latitude, longitude) coordinates.

    Args:
        address: String containing the address to geocode

    Returns:
        Tuple of (latitude, longitude)
    """
    location = geolocator.geocode(address)
    if not location:
        raise ValueError(f"Could not geocode address: {address}")
    return (location.latitude, location.longitude)


def road_distance(coord1, coord2):
    """
    Calculate the road distance between two coordinates.

    Args:
        coord1: Tuple of (latitude, longitude) for the first point
        coord2: Tuple of (latitude, longitude) for the second point

    Returns:
        Distance in meters as a float
    """
    try:
        # OpenRouteService expects coordinates as [longitude, latitude]
        # But our coordinates are (latitude, longitude), so we need to swap
        coords_for_api = [(coord1[1], coord1[0]), (coord2[1], coord2[0])]

        result = client.directions(
            coords_for_api,
            profile='driving-car',
            format='geojson'
        )

        # Extract distance in meters
        distance = result['features'][0]['properties']['summary']['distance']
        return distance
    except Exception as e:
        print(f"[Routing error] {e}")
        return float('inf')  # Return infinity for failed routes


def get_distance_matrix(coords, cache_file="distance_matrix_cache.json"):
    """
    Calculate a matrix of road distances between all locations.
    Includes caching to avoid redundant API calls.

    Args:
        coords: Dictionary mapping location names to (lat, lon) coordinates
        cache_file: File to store/load cached distances

    Returns:
        A 2D list where matrix[i][j] is the road distance from location i to location j
    """
    locations = list(coords.keys())
    n = len(locations)

    # Try to load from cache first
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                cached_locations = cache_data.get("locations", [])
                cached_matrix = cache_data.get("matrix", [])

                # Check if cache matches current locations
                if cached_locations == locations and len(cached_matrix) == n:
                    print(f"Loaded distance matrix from cache: {cache_file}")
                    return cached_matrix
    except Exception as e:
        print(f"Error loading from cache: {e}")

    # Initialize distance matrix
    distance_matrix = [[0] * n for _ in range(n)]

    print(f"Calculating road distances for {n} locations...")

    for i in range(n):
        for j in range(i + 1, n):  # Only calculate upper triangle
            try:
                coord1 = coords[locations[i]]
                coord2 = coords[locations[j]]

                # OpenRouteService expects [longitude, latitude]
                coords_for_api = [(coord1[1], coord1[0]), (coord2[1], coord2[0])]

                response = client.directions(
                    coords_for_api,
                    profile='driving-car',
                    format='geojson'
                )

                # Extract the road distance in meters
                distance = response['features'][0]['properties']['summary']['distance']

                # Store the distance in both directions (symmetric)
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

                print(f"Distance from {locations[i]} to {locations[j]}: {distance:.2f} meters")

                # Add a delay to respect API rate limits
                time.sleep(1.5)

            except Exception as e:
                print(f"Error getting distance from {locations[i]} to {locations[j]}: {e}")
                # Use a large value for failed routes
                distance_matrix[i][j] = distance_matrix[j][i] = float('inf')

    # Save to cache for future use
    try:
        cache_data = {
            "locations": locations,
            "matrix": distance_matrix
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"Saved distance matrix to cache: {cache_file}")
    except Exception as e:
        print(f"Error saving to cache: {e}")

    return distance_matrix


def visualize_route(coordinates, route, output_file="route_map.html"):
    """
    Create an interactive map visualization of the route following actual roads.

    Args:
        coordinates: Dictionary mapping location names to (lat, lon) coordinates
        route: List of location names in the order they should be visited
        output_file: HTML file to save the map to
    """
    try:
        import folium

        # Create map centered at the first location
        first_loc = coordinates[route[0]]
        m = folium.Map(location=first_loc, zoom_start=10)

        # Add markers for each location
        for i, loc_name in enumerate(route):
            if loc_name in coordinates:
                coord = coordinates[loc_name]
                popup_text = f"{i}. {loc_name}"
                folium.Marker(
                    location=coord,
                    popup=popup_text,
                    icon=folium.Icon(color='blue' if i == 0 or i == len(route) - 1 else 'red')
                ).add_to(m)

        # Get actual road paths between consecutive points
        for i in range(len(route) - 1):
            start_name = route[i]
            end_name = route[i + 1]

            if start_name in coordinates and end_name in coordinates:
                try:
                    start_coord = coordinates[start_name]
                    end_coord = coordinates[end_name]

                    # OpenRouteService expects [longitude, latitude]
                    coords_for_api = [(start_coord[1], start_coord[0]), (end_coord[1], end_coord[0])]

                    response = client.directions(
                        coords_for_api,
                        profile='driving-car',
                        format='geojson'
                    )

                    # Extract the route geometry
                    route_coords = []
                    for feature in response['features']:
                        if feature['geometry']['type'] == 'LineString':
                            # Convert [lon, lat] to [lat, lon] for folium
                            for coord in feature['geometry']['coordinates']:
                                route_coords.append([coord[1], coord[0]])

                    # Draw the actual road route
                    folium.PolyLine(
                        route_coords,
                        color='blue',
                        weight=3,
                        opacity=0.7,
                        tooltip=f"Route {start_name} to {end_name}"
                    ).add_to(m)

                    # Add a delay to respect API rate limits
                    time.sleep(1)

                except Exception as e:
                    print(f"Error getting route from {start_name} to {end_name}: {e}")
                    # Fall back to straight line if there's an error
                    folium.PolyLine(
                        [coordinates[start_name], coordinates[end_name]],
                        color='red',
                        weight=2,
                        opacity=0.5,
                        dash_array='5'
                    ).add_to(m)

        # Save the map
        m.save(output_file)
        print(f"Route map saved to {output_file}")

    except ImportError:
        print("Folium not installed. Cannot create visualization.")
        print("Install with: pip install folium")
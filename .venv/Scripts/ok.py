from geopy.geocoders import Nominatim
import folium
import random
from deap import base, creator, tools

# Initialize Geolocator
geolocator = Nominatim(user_agent="ok", timeout=10)

# Define real-world locations (addresses)
locations = {
    'START': 'New York, NY',  # Example: Start location
    'B': 'Los Angeles, CA',
    'C': 'Chicago, IL',
    'D': 'Houston, TX',
    'E': 'Phoenix, AZ'
}

# Geocode addresses to get coordinates (lat, lon)
def geocode_location(address):
    location = geolocator.geocode(address)
    if location:
        return (location.latitude, location.longitude)
    else:
        raise ValueError(f"Could not geocode address: {address}")

# Get coordinates for each location
coordinates = {loc: geocode_location(addr) for loc, addr in locations.items()}

# Function to calculate the distance between two locations
def distance(loc1, loc2):
    lat1, lon1 = coordinates[loc1]
    lat2, lon2 = coordinates[loc2]
    return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5

# Function to evaluate the fitness (total distance of the route)
def evaluate(individual):
    total_distance = 0
    # Map indices back to location names
    locations_ordered = ['START'] + [list(locations.keys())[i+1] for i in individual] + ['START']
    for i in range(len(locations_ordered) - 1):
        total_distance += distance(locations_ordered[i], locations_ordered[i + 1])
    return total_distance,

# Define the fitness class
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize distance

# Define the Individual class
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create the toolbox
toolbox = base.Toolbox()

# Register indices for locations (excluding 'START')
locations_list = list(locations.keys())[1:]  # Exclude 'START'
toolbox.register("indices", random.sample, range(len(locations_list)), len(locations_list))

# Register individual (as indices instead of location names)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

# Register population (list of individuals)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Crossover and mutation work on indices (integers)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

# Selection works as before
toolbox.register("select", tools.selTournament, tournsize=3)

# Evaluate function
toolbox.register("evaluate", evaluate)

# Create initial population
population = toolbox.population(n=100)

# Function to plot the route on Folium map with real coordinates
def plot_route_on_map(route):
    m = folium.Map(location=coordinates['START'], zoom_start=4)  # Center map on starting point
    route_coords = [coordinates['START']] + [coordinates[loc] for loc in route] + [coordinates['START']]

    # Add markers for each location
    for loc, (lat, lon) in coordinates.items():
        folium.Marker([lat, lon], popup=loc).add_to(m)

    # Plot the route as a polyline
    folium.PolyLine(route_coords, color='blue', weight=2.5, opacity=1).add_to(m)
    return m

# Run the genetic algorithm
for generation in range(100):  # Number of generations
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.7:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    for ind in offspring:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    population[:] = offspring

    best_individual = tools.selBest(population, 1)[0]
    print(f"Generation {generation}: Best Route: {best_individual} Fitness: {best_individual.fitness.values[0]}")

# Plot the best route on Folium map
best_individual = tools.selBest(population, 1)[0]
print(f"Best Route: {best_individual} Fitness: {best_individual.fitness.values[0]}")

# Convert indices back to location names for display
best_route_names = ['START'] + [list(locations.keys())[i+1] for i in best_individual] + ['START']

# Save the map as an HTML file
m = plot_route_on_map(best_route_names)
m.save('best_route_map_real_world.html')

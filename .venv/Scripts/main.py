from geo_utils import geocode_location, road_distance, get_distance_matrix, visualize_route
from genetic_optimizer import setup_ga_toolbox
import folium
import random
from deap import tools

# Define your delivery locations
locations = {
    'START': 'New York, NY',
    'B': 'Philadelphia, PA',
    'C': 'Baltimore, MD',
    'D': 'Washington, DC',
    'E': 'Newark, NJ'
}

# Geocode all locations
print("Geocoding addresses...")
coordinates = {k: geocode_location(v) for k, v in locations.items()}
print("Geocoding complete.")

# Getting the distance matrix
print("Getting road distance matrix...")
distance_matrix = get_distance_matrix(coordinates)
print("Matrix ready.")

# Setup GA toolbox
print("Setting up genetic algorithm...")
# Let's adapt to use numerical indices 0, 1, 2, 3 to represent B, C, D, E
location_indices = list(range(len(locations) - 1))  # 0 to 3 (representing B, C, D, E)
toolbox = setup_ga_toolbox(locations, coordinates, road_distance, distance_matrix)

# Initialize the population using the toolbox
print("Running optimization...")
population = toolbox.population(n=100)

# Run Genetic Algorithm
for gen in range(50):
    # Evaluate all individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for i in range(0, len(offspring), 2):
        if random.random() < 0.7 and i + 1 < len(offspring):
            toolbox.mate(offspring[i], offspring[i + 1])
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values

    # Apply mutation on the offspring
    for i in range(len(offspring)):
        if random.random() < 0.2:
            toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    population[:] = offspring

    # Get best individual for this generation
    best_ind = tools.selBest(population, 1)[0]
    print(f"Generation {gen}: Best Distance = {best_ind.fitness.values[0]:.2f}")

# Get final best individual
best_route_indices = tools.selBest(population, 1)[0]

# Convert indices back to location names
# Map the indices 0,1,2,3 to B,C,D,E
location_keys = list(locations.keys())[1:]  # Skip 'START', get B,C,D,E
best_route = [location_keys[idx] for idx in best_route_indices]
best_route_indices = tools.selBest(population, 1)[0]

# Convert indices back to location names
location_keys = list(locations.keys())[1:]  # Skip 'START', get B,C,D,E
best_route = [location_keys[idx] for idx in best_route_indices]

# Add 'START' back to the best route
optimized_route = ['START'] + best_route + ['START']

# Calculate and print total distance
total_distance = toolbox.evaluate(best_route_indices)[0]
print(f"Optimized route: {optimized_route}")
print(f"Total distance: {total_distance:.2f} meters")

# Create and save map with actual road routes
visualize_route(coordinates, optimized_route, "optimized_route_map.html")
print("Map saved as 'optimized_route_map.html'")
# genetic_optimizer.py
from deap import base, creator, tools
import random

# Create a new class for fitness and individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the distance
creator.create("Individual", list, fitness=creator.FitnessMin)


def setup_ga_toolbox(locations, coordinates, road_distance, distance_matrix):
    toolbox = base.Toolbox()

    # Create a list of delivery location indices (excluding 'START')
    # Using indices 0, 1, 2, 3 to represent B, C, D, E
    delivery_indices = list(range(len(locations) - 1))  # 0, 1, 2, 3 for B, C, D, E

    # Register attribute generator
    toolbox.register("indices", random.sample, delivery_indices, len(delivery_indices))

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function using distance_matrix
    def evaluate(individual):
        # Add 1 to each index to skip the START location in the distance matrix
        # If START is at index 0, then B is at 1, C at 2, etc.
        route_with_start = [0]  # START location index
        for idx in individual:
            route_with_start.append(
                idx + 1)  # +1 because indices in individual are 0-based but locations in matrix are 1-based
        route_with_start.append(0)  # Return to START

        total_distance = 0
        for i in range(len(route_with_start) - 1):
            from_idx = route_with_start[i]
            to_idx = route_with_start[i + 1]
            total_distance += distance_matrix[from_idx][to_idx]

        return total_distance,

    # Register genetic operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox
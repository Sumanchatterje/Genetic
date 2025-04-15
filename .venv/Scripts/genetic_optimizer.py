from deap import base, creator, tools
import random

# Avoid redefining the creator if already defined
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the distance
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def setup_ga_toolbox(locations, coordinates, road_distance, distance_matrix, return_to_start=True):
    toolbox = base.Toolbox()

    # Create a list of delivery location indices (excluding 'START')
    delivery_indices = list(range(len(locations) - 1))  # e.g., [0, 1, 2, 3] for 4 delivery points

    # Register attribute generator
    toolbox.register("indices", random.sample, delivery_indices, len(delivery_indices))

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function using distance_matrix
    def evaluate(individual):
        route_with_start = [0]  # Index 0 is always 'START'
        route_with_start += [idx + 1 for idx in individual]  # +1 for correct matrix lookup
        if return_to_start:
            route_with_start.append(0)  # Return to START only if checkbox checked

        total_distance = 0
        for i in range(len(route_with_start) - 1):
            from_idx = route_with_start[i]
            to_idx = route_with_start[i + 1]
            total_distance += distance_matrix[from_idx][to_idx]

        return total_distance,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

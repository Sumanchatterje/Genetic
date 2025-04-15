import random
from deap import base, creator, tools
from geo_utils import osrm_route_distance

def evaluate(individual, coordinates):
    total_distance = 0
    current = 'START'
    for loc in individual:
        total_distance += osrm_route_distance(coordinates[current], coordinates[loc])
        current = loc
    total_distance += osrm_route_distance(coordinates[current], coordinates['START'])  # return to start
    return total_distance,

def setup_ga(location_keys, coordinates):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, location_keys, len(location_keys))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    return toolbox

from flask import Flask, render_template, request
from geo_utils import geocode_location, road_distance, get_distance_matrix, visualize_route
from genetic_optimizer import setup_ga_toolbox
from deap import tools
import random

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get starting location
        locations = {'START': request.form['start']}

        # Dynamically collect delivery points
        count = 1
        while True:
            loc = request.form.get(f'point{count}')
            if not loc:
                break
            locations[f'P{count}'] = loc
            count += 1

        if len(locations) <= 1:
            return render_template("index.html", error="Please enter at least one delivery point.")

        # Geocode locations
        coordinates = {k: geocode_location(v) for k, v in locations.items()}
        distance_matrix = get_distance_matrix(coordinates)

        # Setup GA
        toolbox = setup_ga_toolbox(locations, coordinates, road_distance, distance_matrix)
        population = toolbox.population(n=100)

        # Run GA
        for _ in range(50):
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            for i in range(0, len(offspring), 2):
                if random.random() < 0.7 and i + 1 < len(offspring):
                    toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values
            for i in range(len(offspring)):
                if random.random() < 0.2:
                    toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population[:] = offspring

        best_indices = tools.selBest(population, 1)[0]
        location_keys = list(locations.keys())[1:]  # Skip START
        best_route = ['START'] + [location_keys[i] for i in best_indices] + ['START']
        total_distance = toolbox.evaluate(best_indices)[0]

        # Visualize
        visualize_route(coordinates, best_route, "static/optimized_route_map.html")

        return render_template("result.html", route=best_route, distance=total_distance)

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
